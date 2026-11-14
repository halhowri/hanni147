"""
Quantization experiment: the SAME fine-tuned model (version 2) stored at different precisions.
For each one: size, memory, speed and quality on the 94 test rows.

    ../llm_env/bin/python quant_experiment.py build      # make the variants as Ollama models
    ../llm_env/bin/python quant_experiment.py measure    # run the test set through each
    ../llm_env/bin/python quant_experiment.py report     # print the table again

Needs outputs/merged/ (from export_ollama.py), ../tools/llama.cpp with build/bin/llama-quantize,
and the Ollama model tf2torch:v1 (template for the file layout).
"""
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
WORK = os.path.join(HERE, "..", "quant_work")            # large temporary files, outside the repository
LLAMA = os.path.join(HERE, "..", "tools", "llama.cpp")
LEVELS = ["F16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q3_K_M", "Q2_K"]     # most to fewest bits per weight
OLLAMA = "http://localhost:11434"
name_of = lambda level: "tf2torch-" + level.lower().replace("_", "")


def build():
    os.makedirs(WORK, exist_ok=True)
    f16 = os.path.join(WORK, "upstream_F16.gguf")
    if not os.path.exists(f16):
        subprocess.run([sys.executable, os.path.join(LLAMA, "convert_hf_to_gguf.py"), os.path.join(OUT, "merged"),
                        "--outfile", f16, "--outtype", "f16"], check=True)
    shown = subprocess.run(["ollama", "show", "tf2torch:v1", "--modelfile"], check=True, capture_output=True, text=True).stdout
    template = next(l.split(" ", 1)[1].strip() for l in shown.splitlines() if l.startswith("FROM "))
    modelfile = open(os.path.join(OUT, "Modelfile")).read().split("\n", 1)[1]        # everything after FROM
    sizes = {}
    for level in LEVELS:
        up = os.path.join(WORK, "upstream_%s.gguf" % level)
        if level != "F16":
            subprocess.run([os.path.join(LLAMA, "build", "bin", "llama-quantize"), f16, up, level], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        final = os.path.join(WORK, "ollama_%s.gguf" % level)
        subprocess.run([sys.executable, os.path.join(HERE, "repack_gguf_for_ollama.py"), template, up, final], check=True)
        mf = os.path.join(WORK, "Modelfile_%s" % level)
        open(mf, "w").write("FROM %s\n%s" % (final, modelfile))
        subprocess.run(["ollama", "create", name_of(level), "-f", mf], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        sizes[level] = dict(file_mb=os.path.getsize(final) / 1e6, language_part_mb=os.path.getsize(up) / 1e6)
        os.remove(final)
        if level != "F16":
            os.remove(up)
        print("built", name_of(level), "%.0f MB" % sizes[level]["file_mb"], flush=True)
    json.dump(sizes, open(os.path.join(OUT, "quant_sizes.json"), "w"), indent=1)


def ask(model, text):
    r = requests.post(OLLAMA + "/api/chat", timeout=900, json={
        "model": model, "stream": False, "think": False, "messages": [{"role": "user", "content": text}],
        "options": {"temperature": 0, "num_predict": 640}})
    if r.status_code != 200:      # e.g. Ollama stops a model that repeats one token forever
        return {"message": {"content": ""}, "failed": r.text[:200]}
    return r.json()


def measure():
    sys.path.insert(0, HERE)
    from evaluate import load_test, score_one
    tests = load_test()
    reference = json.load(open(os.path.join(OUT, "answers_tuned.json")))      # the adapter in Transformers
    sizes = json.load(open(os.path.join(OUT, "quant_sizes.json")))
    path = os.path.join(OUT, "quant_results.json")
    results = json.load(open(path)) if os.path.exists(path) else {}
    for level in (sys.argv[2:] or LEVELS):          # optional: only the levels named on the command line
        model = name_of(level)
        subprocess.run(["ollama", "stop", model], capture_output=True)
        first = ask(model, tests[0]["messages"][1]["content"])                # includes loading the model
        ps = next(m for m in requests.get(OLLAMA + "/api/ps").json()["models"] if m["name"].startswith(model))
        answers, aborted, tok, gen_s, prompt_tok, prompt_s, wall = {}, 0, 0, 0.0, 0, 0.0, time.time()
        for t in tests:
            j = ask(model, t["messages"][1]["content"])
            answers[t["id"]] = j["message"]["content"]
            aborted += "failed" in j
            tok += j.get("eval_count", 0); gen_s += j.get("eval_duration", 0) / 1e9
            prompt_tok += j.get("prompt_eval_count", 0); prompt_s += j.get("prompt_eval_duration", 0) / 1e9
        wall = time.time() - wall
        subprocess.run(["ollama", "stop", model], capture_output=True)
        with ThreadPoolExecutor(8) as pool:
            rows = list(pool.map(score_one, [(t, answers[t["id"]]) for t in tests]))
        runnable = [r for r in rows if r["runs"] is not None]
        results[level] = dict(
            model=model, **sizes[level], memory_mb=ps["size"] / 1e6, gpu_share=ps["size_vram"] / ps["size"],
            load_s=first.get("load_duration", 0) / 1e9, tokens_per_s=tok / max(gen_s, 1e-9), aborted=aborted, prompt_tokens_per_s=prompt_tok / max(prompt_s, 1e-9),
            seconds_per_answer=wall / len(tests),
            valid=sum(r["valid"] for r in rows) / len(rows), runs=sum(bool(r["runs"]) for r in runnable) / len(runnable),
            n_runnable=len(runnable), exact=sum(r["exact"] for r in rows) / len(rows),
            same_as_full_precision=sum(answers[t["id"]].strip() == reference[t["id"]].strip() for t in tests) / len(tests),
            groups={g: sum(bool(r["runs"]) for r in runnable if r["group"] == g) / max(sum(r["group"] == g for r in runnable), 1)
                    for g in ("seen_patterns", "unseen_patterns", "real_code")})
        json.dump(results, open(os.path.join(OUT, "quant_results.json"), "w"), indent=1)
        print("measured", level, flush=True)
    report()


def report():
    res = json.load(open(os.path.join(OUT, "quant_results.json")))
    print("\n%-8s %8s %9s %5s %7s %8s %9s | %7s %7s %7s %10s" % ("level", "file MB", "memory MB", "GPU", "load s", "tok/s", "s/answer",
                                                                 "valid", "runs", "exact", "same as tf"))
    for level in [l for l in LEVELS if l in res]:
        r = res[level]
        print("%-8s %8.0f %9.0f %4.0f%% %7.1f %8.0f %9.2f | %6.0f%% %6.0f%% %6.0f%% %9.0f%%" % (
            level, r["file_mb"], r["memory_mb"], 100 * r["gpu_share"], r["load_s"], r["tokens_per_s"], r["seconds_per_answer"],
            100 * r["valid"], 100 * r["runs"], 100 * r["exact"], 100 * r["same_as_full_precision"]))
    print("runs = executed and gave the reference numbers (of %d checkable rows); same as tf = identical text to the\n"
          "adapter's answer in Transformers. Every file includes the same ~%.0f MB vision part."
          % (next(iter(res.values()))["n_runnable"], next(iter(res.values()))["file_mb"] - next(iter(res.values()))["language_part_mb"]))


if __name__ == "__main__":
    {"build": build, "measure": measure, "report": report}[sys.argv[1] if len(sys.argv) > 1 else "report"]()
