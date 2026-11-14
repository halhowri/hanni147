"""
Step 4 - put the fine-tuned model into Ollama.

1. merge the LoRA add-on into the base model  -> outputs/merged/
2. write a Modelfile                          -> outputs/Modelfile
3. ollama create tf2torch ... (quantised to q4_K_M)

Run:   ../llm_env/bin/python export_ollama.py
Then:  ollama run tf2torch
"""
import os
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "outputs")
MERGED = os.path.join(OUT, "merged")
OLLAMA_NAME = "tf2torch"
SYSTEM = ("You convert TensorFlow 1.x code to PyTorch. Keep the same variable "
          "names. Reply with only the PyTorch code.")

if not os.path.exists(os.path.join(MERGED, "config.json")):
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=os.path.join(OUT, "lora"), max_seq_length=2048,
                                                         load_in_4bit=False, load_in_16bit=True)
    model.save_pretrained_merged(MERGED, tokenizer, save_method="merged_16bit")

modelfile = os.path.join(OUT, "Modelfile")
with open(modelfile, "w") as f:
    f.write(f"FROM {MERGED}\n"
            "TEMPLATE {{ .Prompt }}\n"
            "RENDERER qwen3.5\n"
            "PARSER qwen3.5\n"
            f'SYSTEM """{SYSTEM}"""\n'
            "PARAMETER temperature 0\n"          # same as the greedy decoding used in evaluate.py
            "PARAMETER repeat_penalty 1\n"
            "PARAMETER num_ctx 2048\n")

subprocess.run(["ollama", "create", OLLAMA_NAME, "-f", modelfile, "--quantize", "q4_K_M"], check=True)
print(f"\nDone. Try it with:  ollama run {OLLAMA_NAME}")
