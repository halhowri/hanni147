"""
Runs a piece of PyTorch code in a separate, CPU-only Python process and
returns the values of chosen variables, so that a model's translation can be
compared with the reference translation by RESULT, not just by text.

Used by build_dataset.py (to prove every reference answer really runs) and by
evaluate.py (to score the base model and the fine-tuned model).
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HEADER = """
import json
import numpy as np
import torch
torch.set_num_threads(1)
torch.manual_seed(0)
rs = np.random.RandomState(0)
"""

FOOTER = """
def _ser(v):
    if isinstance(v, torch.Tensor):
        v = v.detach().cpu().double().numpy()
    if isinstance(v, torch.Size):
        v = list(v)
    if isinstance(v, str):
        return {"str": v}
    if isinstance(v, (list, tuple)):
        return {"list": [_ser(i) for i in v]}
    a = np.asarray(v, dtype=float)
    return {"shape": list(a.shape), "data": a.flatten().tolist()}

print("@@RESULT@@" + json.dumps({k: _ser(globals()[k]) for k in %r}))
"""


def run_snippet(setup, code, post, outs, timeout=60):
    """Returns (values_dict, None) on success or (None, error_text)."""
    script = HEADER + "\n" + setup + "\n" + code + "\n" + (post or "") + "\n" + FOOTER % (outs,)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snippet.py")
        with open(path, "w") as f:
            f.write(script)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
        try:
            p = subprocess.run([sys.executable, "-I", path], cwd=tmp, env=env,
                               capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return None, "timeout"
    for line in p.stdout.splitlines():
        if line.startswith("@@RESULT@@"):
            return json.loads(line[len("@@RESULT@@"):]), None
    err = (p.stderr.strip().splitlines() or ["no output"])[-1]
    return None, err[:200]


def same(a, b, mode="value"):
    """Compare two serialised values. mode='shape' ignores the numbers
    (used when the code draws random numbers)."""
    if "str" in a or "str" in b:
        return a == b
    if "list" in a or "list" in b:
        if "list" not in a or "list" not in b or len(a["list"]) != len(b["list"]):
            return False
        return all(same(x, y, mode) for x, y in zip(a["list"], b["list"]))
    sa = [d for d in a["shape"]] or [1]
    sb = [d for d in b["shape"]] or [1]
    if sa != sb:
        return False
    if mode == "shape":
        return True
    return bool(np.allclose(a["data"], b["data"], rtol=1e-3, atol=1e-5, equal_nan=True))


def check(example, code):
    """True/False if the example can be checked by running it, else None."""
    if not example.get("outs"):
        return None
    ref, err = run_snippet(example["setup"], example["pt"], example.get("post"), example["outs"])
    if ref is None:
        raise RuntimeError(f"reference failed for {example['id']}: {err}")
    got, err = run_snippet(example["setup"], code, example.get("post"), example["outs"])
    if got is None:
        return False
    mode = example.get("mode", "value")
    return all(k in got and same(ref[k], got[k], mode) for k in example["outs"])
