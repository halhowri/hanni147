"""
Step 4 - put the fine-tuned model into Ollama.

1. merge the LoRA add-on into the base model          -> outputs/merged/
2. convert the merged model to a GGUF file with llama.cpp's converter, then quantise it (QUANT)
3. repack that file into the tensor layout Ollama's runner expects (repack_gguf_for_ollama.py),
   using an existing Ollama Qwen3.5 model as the template
4. write a Modelfile and run: ollama create tf2torch

Version 1 was made with Ollama 0.23, which imported outputs/merged/ directly and quantised it to
q4_K_M. Ollama 0.40 cannot do that on this laptop (it needs the MLX runtime), hence steps 2 and 3.
The weights are quantised to 6-bit (Q6_K, about 0.85 GB) with llama.cpp's llama-quantize:
quant_experiment.py showed this halves memory and doubles speed with no loss on the trained patterns.

Needs:  llama.cpp cloned in ../tools/llama.cpp   (git clone --depth 1 https://github.com/ggml-org/llama.cpp)
        its llama-quantize tool built in ../tools/llama.cpp/build/bin   (cmake --build ... --target llama-quantize)
        pip install ../tools/llama.cpp/gguf-py   (into ../llm_env)
        the Ollama model TEMPLATE_MODEL below    (version 1 of this model, kept as tf2torch:v1)

Run:   rm -rf outputs/merged      (after retraining, so the merge is done again)
       ../llm_env/bin/python export_ollama.py
Then:  ollama run tf2torch
"""
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
MERGED = os.path.join(OUT, "merged")
OLLAMA_NAME = "tf2torch"
TEMPLATE_MODEL = "tf2torch:v1"      # an Ollama Qwen3.5-0.8B model that already loads
QUANT = "Q6_K"                      # precision of the weights; chosen with quant_experiment.py
QUANTIZE = os.path.join(HERE, "..", "tools", "llama.cpp", "build", "bin", "llama-quantize")
CONVERTER = os.path.join(HERE, "..", "tools", "llama.cpp", "convert_hf_to_gguf.py")
SYSTEM = ("You convert TensorFlow 1.x code to PyTorch. Keep the same variable "
          "names. Reply with only the PyTorch code.")

if not os.path.exists(os.path.join(MERGED, "config.json")):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=os.path.join(OUT, "lora"), max_seq_length=2048,
                                                         load_in_4bit=False, load_in_16bit=True)
    model.save_pretrained_merged(MERGED, tokenizer, save_method="merged_16bit")

# ---- GGUF in Ollama's layout --------------------------------------------------------------
upstream = os.path.join(OUT, "upstream_f16.gguf")
final = os.path.join(OUT, "tf2torch.gguf")
subprocess.run([sys.executable, CONVERTER, MERGED, "--outfile", upstream, "--outtype", "f16"], check=True)
shown = subprocess.run(["ollama", "show", TEMPLATE_MODEL, "--modelfile"], check=True,
                       capture_output=True, text=True).stdout
template = next(line.split(" ", 1)[1].strip() for line in shown.splitlines() if line.startswith("FROM "))
quantized = os.path.join(OUT, "upstream_quant.gguf")
subprocess.run([QUANTIZE, upstream, quantized, QUANT], check=True)
subprocess.run([sys.executable, os.path.join(HERE, "repack_gguf_for_ollama.py"), template, quantized, final],
               check=True)
os.remove(quantized)

modelfile = os.path.join(OUT, "Modelfile")
with open(modelfile, "w") as f:
    f.write(f"FROM {final}\n"
            "TEMPLATE {{ .Prompt }}\n"
            "RENDERER qwen3.5\n"
            "PARSER qwen3.5\n"
            f'SYSTEM """{SYSTEM}"""\n'
            "PARAMETER temperature 0\n"          # same as the greedy decoding used in evaluate.py
            "PARAMETER repeat_penalty 1\n"
            "PARAMETER num_ctx 2048\n")

subprocess.run(["ollama", "create", OLLAMA_NAME, "-f", modelfile], check=True)
os.remove(upstream)
os.remove(final)                      # Ollama keeps its own copy
print(f"\nDone. Try it with:  ollama run {OLLAMA_NAME}")
