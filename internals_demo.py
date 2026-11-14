"""
Looking INSIDE the model to explain one failure of the fine-tuned TF->PyTorch model.

Failure (test example unseen_patterns-dropout-0):
    TensorFlow : l1 = tf.nn.dropout(l1, keep_prob=0.9)
    correct    : l1 = torch.nn.functional.dropout(l1, p=0.1)
    tuned model: l1 = torch.dropout(l1, keep_prob=0.9)

Run:  ../llm_env/bin/python internals_demo.py base     then     ... internals_demo.py tuned
Three views, each at the exact position where the answer goes wrong:
  1. next-token probabilities   what did the model want to write, and how sure was it?
  2. logit lens                 read the prediction out of EVERY layer: where is the choice made?
  3. attention                  which input tokens was the model looking at when it chose?
"""
import json, os, sys
import torch
from unsloth import FastLanguageModel

HERE = os.path.dirname(os.path.abspath(__file__))
which = sys.argv[1]
name = "unsloth/Qwen3.5-0.8B" if which == "base" else os.path.join(HERE, "outputs", "lora")
model, tok = FastLanguageModel.from_pretrained(model_name=name, max_seq_length=2048, load_in_4bit=False, load_in_16bit=True)
FastLanguageModel.for_inference(model)

test = [json.loads(l) for l in open(os.path.join(HERE, "data", "test.jsonl"))]
ex = next(t for t in test if t["id"] == "unseen_patterns-dropout-0")
prompt = tok.apply_chat_template(ex["messages"][:2], tokenize=False, add_generation_prompt=True, enable_thinking=False)
line1 = "l1 = torch.relu(torch.matmul(xs, W1) + b1)\n"
REF = line1 + "l1 = torch.nn.functional.dropout(l1, p=0.1)"
GOT = line1 + "l1 = torch.dropout(l1, keep_prob=0.9)"
POINTS = [  # (label, one full answer, another full answer) -> examined at the first token where they differ
    ("A: which function?", REF, GOT),
    ("B: which argument? (after the wrong function)", line1 + "l1 = torch.dropout(l1, p=0.1)", GOT),
    ("C: which argument? (after the RIGHT function)", REF, line1 + "l1 = torch.nn.functional.dropout(l1, keep_prob=0.9)"),
]
ids = lambda text: tok(text=[text], return_tensors="pt")["input_ids"][0]
head = model.get_output_embeddings()
norms = [m for n, m in model.named_modules() if n.endswith("norm") and "layers" not in n and "vis" not in n]
final_norm = norms[-1]

try:
    model.set_attn_implementation("eager")       # needed to get attention weights back
except Exception as e:
    print("(attention weights not available: %s)" % e)

for label, good_text, bad_text in POINTS:
    g, w = ids(prompt + good_text), ids(prompt + bad_text)
    d = next(i for i in range(min(len(g), len(w))) if g[i] != w[i])     # first differing token
    x, good_id, bad_id = g[:d], int(g[d]), int(w[d])
    print("\n    answer so far: ...%r" % tok.decode(x[-12:]))
    with torch.no_grad():
        out = model(input_ids=x[None].to("cuda"), output_hidden_states=True, output_attentions=True)
    p = out.logits[0, -1].float().softmax(-1)
    top = p.topk(5)
    print("\n=== %s model, point %s" % (which, label))
    print("1. next token: P(correct %r) = %.3f   P(tuned's %r) = %.3f   top-5: %s" % (
        tok.decode([good_id]), p[good_id], tok.decode([bad_id]), p[bad_id],
        ", ".join("%r %.2f" % (tok.decode([int(i)]), v) for v, i in zip(top.values, top.indices))))
    print("2. logit lens (prediction read out of each layer):")
    hs = out.hidden_states
    for layer in range(len(hs) - 9, len(hs)):
        with torch.no_grad():
            pl = head(final_norm(hs[layer][0, -1])).float().softmax(-1)
        print("   layer %2d: P(correct) %.3f  P(tuned's) %.3f  top: %r" % (layer, pl[good_id], pl[bad_id], tok.decode([int(pl.argmax())])))
    att = out.attentions
    if att and att[0] is not None:
        a = torch.stack([l[0, :, -1, :].float().mean(0) for l in att[len(att) // 2:]]).mean(0).cpu()   # upper half of the layers
        start = len(ids(prompt.split("<|im_start|>user")[0]))          # skip the system prompt
        pieces = [(float(a[i]), i) for i in range(start, len(x))]
        print("3. attention from this position (mean of upper layers), strongest input tokens:")
        print("   " + " | ".join("%r %.3f" % (tok.decode([int(x[i])]), v) for v, i in sorted(pieces, reverse=True)[:10]))
    else:
        print("3. attention not returned by this model build")
