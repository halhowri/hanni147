"""
Put new weights into the GGUF layout that Ollama's own runner expects for Qwen3.5.

Why: Ollama 0.40 can no longer import the merged safetensors folder on this laptop, and the GGUF
written by llama.cpp's convert_hf_to_gguf.py uses a different tensor layout from Ollama's
(e.g. one 'attn_qkv_gate' tensor where Ollama wants 'attn_qkv' + 'attn_gate').

How: take a GGUF that Ollama already loads (the version 1 model file) as the TEMPLATE -- all its
settings and tokenizer are kept -- and replace the language model's weights with the new ones:
    same name                       -> new weights
    attn_qkv / attn_gate            -> first 6144 / last 2048 rows of attn_qkv_gate
    ssm_beta / ssm_alpha            -> first 16 / last 16 rows of ssm_beta_alpha
    ssm_dt                          -> ssm_dt.bias
    vision tower (v.*) and mtp.*    -> kept from the template (not used for text-only translation)
New weights are stored as 16-bit floats (no quantisation loss).

Usage: python repack_gguf_for_ollama.py TEMPLATE.gguf NEW_UPSTREAM_F16.gguf OUT.gguf
"""
import sys

import numpy as np
import gguf

template = gguf.GGUFReader(sys.argv[1])
new = {t.name: t for t in gguf.GGUFReader(sys.argv[2]).tensors}
F16, F32 = gguf.GGMLQuantizationType.F16, gguf.GGMLQuantizationType.F32


def new_weights(name):
    """New weights for a template tensor name, or None if the new file has no such weights."""
    if name in new:
        return np.array(new[name].data)
    if name.endswith(".attn_qkv.weight"):
        return np.array(new[name.replace("attn_qkv", "attn_qkv_gate")].data)[:6144]
    if name.endswith(".attn_gate.weight"):
        return np.array(new[name.replace("attn_gate", "attn_qkv_gate")].data)[6144:]
    if name.endswith(".ssm_beta.weight"):
        return np.array(new[name.replace("ssm_beta", "ssm_beta_alpha")].data)[:16]
    if name.endswith(".ssm_alpha.weight"):
        return np.array(new[name.replace("ssm_alpha", "ssm_beta_alpha")].data)[16:]
    if name.endswith(".ssm_dt"):
        return np.array(new[name + ".bias"].data)
    return None


arch = template.fields[gguf.Keys.General.ARCHITECTURE].contents()
writer = gguf.GGUFWriter(sys.argv[3], arch=arch)
for field in template.fields.values():
    if field.name == gguf.Keys.General.ARCHITECTURE or field.name.startswith("GGUF."):
        continue
    val_type = field.types[0]
    sub_type = field.types[-1] if val_type == gguf.GGUFValueType.ARRAY else None
    value = field.contents()
    if val_type == gguf.GGUFValueType.ARRAY and len(value) == 0:
        print("left out (empty list, the library cannot write it):", field.name)
        continue
    writer.add_key_value(field.name, value, val_type, sub_type=sub_type)

payload, replaced, kept = [], 0, []
for t in template.tensors:
    shape = tuple(int(d) for d in reversed(t.shape))          # numpy order
    w = new_weights(t.name)
    if w is None:
        kept.append(t.name)
        if t.name.startswith(("blk.", "token_embd", "output_norm")):
            sys.exit("no new weights for language-model tensor %s" % t.name)
        # kept as stored (for quantised tensors the library wants the stored byte shape)
        writer.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
        payload.append(t.data)
        continue
    if w.shape != shape:
        sys.exit("%s: new shape %s, template shape %s" % (t.name, w.shape, shape))
    keep32 = t.tensor_type == F32
    w = np.ascontiguousarray(w.astype(np.float32 if keep32 else np.float16))
    writer.add_tensor_info(t.name, shape, w.dtype, w.nbytes, F32 if keep32 else F16)
    payload.append(w)
    replaced += 1

writer.write_header_to_file()
writer.write_kv_data_to_file()
writer.write_ti_data_to_file()
for data in payload:
    writer.write_tensor_data(data)
writer.close()
groups = sorted(set(n.split(".")[0] for n in kept))
print("replaced %d tensors with the new weights; kept %d from the template (%s)" % (replaced, len(kept), ", ".join(groups)))
