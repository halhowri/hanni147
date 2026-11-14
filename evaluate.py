"""
Step 3 - compare the ORIGINAL Qwen3.5-0.8B with the FINE-TUNED one on the
test set (data/test.jsonl), which was never used in training.

Run, in this order:
    ../llm_env/bin/python evaluate.py generate base
    ../llm_env/bin/python evaluate.py generate tuned
    ../llm_env/bin/python evaluate.py score

Scores, per group of test examples:
    valid Python   the answer is syntactically correct Python
    no TF left     the answer no longer mentions tf. / tensorflow
    runs correctly the answer is EXECUTED and gives the same numbers as the
                   reference PyTorch code (the strictest and most meaningful score)
    exact match    identical to the reference, ignoring blank lines / spacing at line ends
    similarity     0-100, how close the text is to the reference
"""
import ast
import difflib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
MODEL_NAME = "unsloth/Qwen3.5-0.8B"
GROUPS = ["seen_patterns", "unseen_patterns", "real_code"]


def load_test():
    with open(os.path.join(HERE, "data", "test.jsonl")) as f:
        return [json.loads(line) for line in f]


def clean(text):
    """Keep only the code: drop any <think> block and Markdown code fences."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    fenced = re.findall(r"```(?:python|py)?\n(.*?)```", text, flags=re.S)
    if fenced:
        text = "\n".join(fenced)
    return text.strip("\n")


def generate(which):
    from unsloth import FastLanguageModel
    import torch
    name = MODEL_NAME if which == "base" else os.path.join(OUT, "lora")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=name, max_seq_length=2048,
                                                         load_in_4bit=False, load_in_16bit=True)
    FastLanguageModel.for_inference(model)
    tokenizer.padding_side = "left"
    tests = load_test()
    prompts = [tokenizer.apply_chat_template(t["messages"][:2], tokenize=False, add_generation_prompt=True,
                                             enable_thinking=False) for t in tests]
    order = sorted(range(len(tests)), key=lambda i: len(prompts[i]))
    answers = {}
    for start in range(0, len(order), 4):
        idx = order[start:start + 4]
        # the Qwen3.5 "tokenizer" is a text+image processor, so the text must be passed by name
        batch = tokenizer(text=[prompts[i] for i in idx], return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            out = model.generate(**batch, max_new_tokens=640, do_sample=False)   # greedy = repeatable
        for i, row in zip(idx, out):
            answers[tests[i]["id"]] = tokenizer.decode(row[batch["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"{which}: {min(start + 4, len(order))}/{len(order)}", flush=True)
    with open(os.path.join(OUT, f"answers_{which}.json"), "w") as f:
        json.dump(answers, f, indent=1)


def norm(code):
    return "\n".join(l.rstrip() for l in code.strip().splitlines() if l.strip())


def score_one(args):
    from exec_check import check
    test, raw = args
    code = clean(raw)
    try:
        ast.parse(code)
        valid = bool(code.strip())
    except SyntaxError:
        valid = False
    return dict(
        id=test["id"], group=test["group"], category=test["category"], valid=valid,
        no_tf=not re.search(r"\btf\.|tensorflow", code),
        exact=norm(code) == norm(test["pt"]),
        similarity=100 * difflib.SequenceMatcher(None, norm(code), norm(test["pt"])).ratio(),
        runs=(check(test, code) if valid else (False if test["outs"] else None)),
        answer=code,
    )


def score():
    tests = load_test()
    report, table = {}, []
    for which in ["base", "tuned"]:
        with open(os.path.join(OUT, f"answers_{which}.json")) as f:
            answers = json.load(f)
        with ThreadPoolExecutor(8) as pool:
            rows = list(pool.map(score_one, [(t, answers[t["id"]]) for t in tests]))
        report[which] = rows
        for group in GROUPS + ["ALL"]:
            sel = [r for r in rows if group == "ALL" or r["group"] == group]
            runnable = [r for r in sel if r["runs"] is not None]
            pct = lambda xs: 100 * sum(xs) / max(len(xs), 1)
            table.append(dict(model=which, group=group, n=len(sel),
                              valid=pct([r["valid"] for r in sel]), no_tf=pct([r["no_tf"] for r in sel]),
                              runs=pct([r["runs"] for r in runnable]), n_runnable=len(runnable),
                              exact=pct([r["exact"] for r in sel]),
                              similarity=sum(r["similarity"] for r in sel) / max(len(sel), 1)))
    with open(os.path.join(OUT, "comparison.json"), "w") as f:
        json.dump({"summary": table, "details": report, "tests": {t["id"]: t for t in tests}}, f, indent=1)

    head = f"{'group':<16}{'model':<7}{'n':>4}{'valid Python':>14}{'no TF left':>12}{'runs correctly':>16}{'exact match':>13}{'similarity':>12}"
    print(head)
    print("-" * len(head))
    for group in GROUPS + ["ALL"]:
        for row in [r for r in table if r["group"] == group]:
            print(f"{row['group']:<16}{row['model']:<7}{row['n']:>4}{row['valid']:>13.0f}%{row['no_tf']:>11.0f}%"
                  f"{row['runs']:>10.0f}% of {row['n_runnable']:<2}{row['exact']:>12.0f}%{row['similarity']:>12.0f}")
    print("\nfull details: outputs/comparison.json")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "generate" and sys.argv[2] in ("base", "tuned"):
        generate(sys.argv[2])
    elif len(sys.argv) == 2 and sys.argv[1] == "score":
        score()
    else:
        sys.exit(__doc__)
