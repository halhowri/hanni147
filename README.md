# Fine-tuning Qwen3.5-0.8B to translate TensorFlow 1.x code to PyTorch

A small LoRA fine-tune done on a laptop GPU (RTX 3050 Ti, 4 GB) with [Unsloth](https://github.com/unslothai/unsloth).
The model learns to rewrite TensorFlow 1.x code (placeholders, `tf.Variable`, `tf.Session`, `feed_dict`,
`AdamOptimizer`...) as PyTorch.

## Steps

| Step | Script | What it does |
|---|---|---|
| 1 | `build_dataset.py` | Builds TensorFlow -> PyTorch pairs: 906 train, 100 validation, 94 test. Every reference answer is executed to prove it runs. |
| 2 | `train.py` | LoRA fine-tune: 16-bit, rank 16, batch size 1, gradient accumulation 4, max sequence length 512, 3 epochs (about 8 minutes). |
| 3 | `evaluate.py` | Runs the original and the fine-tuned model on the test set and scores both. |
| 4 | `export_ollama.py` | Merges the LoRA add-on, converts it to GGUF, quantises it to Q6_K and creates the Ollama model `tf2torch`. See the Ollama note under Version 2. |
| extra | `internals_demo.py` | Looks inside the model at the token where an answer goes wrong (next-token probabilities, logit lens, attention). |

`exec_check.py` runs generated code in a separate CPU process and compares its numbers with the reference.

## Results on the test set (never used in training)

"Runs correctly" means the answer was executed and produced the same numbers as the reference PyTorch code.

| Test group | Model | Valid Python | No TensorFlow left | Runs correctly | Exact match |
|---|---|---|---|---|---|
| Seen patterns (65) | original | 95% | 40% | 4% of 50 | 0% |
| Seen patterns (65) | fine-tuned | 100% | 100% | 100% of 50 | 100% |
| Unseen patterns (21) | original | 95% | 62% | 10% of 21 | 0% |
| Unseen patterns (21) | fine-tuned | 100% | 100% | 67% of 21 | 57% |
| Real project code (8) | original | 100% | 38% | 0% of 6 | 0% |
| Real project code (8) | fine-tuned | 100% | 100% | 83% of 6 | 25% |
| All (94) | original | 96% | 45% | 5% of 77 | 0% |
| All (94) | fine-tuned | 100% | 100% | 90% of 77 | 84% |

Validation loss: 0.0026 -> 0.0006 -> 0.0002 over the three epochs.

These are the version 2 numbers. Since version 2, dropout is part of the training data, so the three
dropout rows in "unseen patterns" are held-out values of a trained pattern, not a pattern the model never saw.

## Version 2: a failure explained from inside the model, then fixed

Version 1 translated `tf.nn.dropout(l1, keep_prob=0.9)` to `torch.dropout(l1, keep_prob=0.9)`
(correct: `torch.nn.functional.dropout(l1, p=0.1)`). `internals_demo.py` showed why:

- After `l1 = torch`, version 1 gave the wrong token `.dropout` 99.7% and the correct `.nn` 0.1%
  (the original model: 89% and 8%).
- When handed the right function name, version 1 chose the right argument `p` with 94%. The whole
  failure was one token.
- 53% of the attention at that position went to the source token `.dropout`: the model was copying the name.

Cause: nearly every training pair kept the function name (`tf.X` -> `torch.X`). Version 2 adds 120
training pairs where the PyTorch name must change (50 dropout, 70 other renames). The test set is unchanged.

| Test rows | Version 1 runs correctly | Version 2 runs correctly |
|---|---|---|
| Seen patterns | 100% of 50 | 100% of 50 |
| Unseen patterns | 48% of 21 | 67% of 21 |
| Real project code | 67% of 6 | 83% of 6 |
| All | 83% of 77 | 90% of 77 |
| Dropout rows | 0 of 3 | 3 of 3 |
| L2 loss rows (kept out of training) | 0 of 3 | 0 of 3 |
| Cross-entropy row (kept out of training) | 0 of 1 | 0 of 1 |

No row that version 1 got right is wrong in version 2. After retraining, the correct `.nn` gets 100% at
the same position. The L2 and cross-entropy rows show the model learned the renames it was shown, not a
general rule.

Ollama: `tf2torch` is now version 2 (`tf2torch:v1` keeps version 1). Ollama 0.40 can no longer import
the merged safetensors folder on this laptop (it needs the MLX runtime), so `export_ollama.py` now converts
the merged model to GGUF with llama.cpp's converter and `repack_gguf_for_ollama.py` rearranges it into the
tensor layout Ollama's runner expects. Checked on all 94 test inputs at 16-bit: the Ollama model gives the
same answer as the adapter in Transformers on 93. The vision part of the model file is carried over from
version 1 unchanged; the model is meant for text only.

## Quantization experiment

`quant_experiment.py` stores the same version 2 model at seven precisions and runs the 94 test rows
through each in Ollama (RTX 3050 Ti, everything on the GPU).

| Precision | File | GPU memory | Tokens/s | Runs correctly (of 77) | Same text as 16-bit adapter |
|---|---|---|---|---|---|
| F16 | 1732 MB | 1757 MB | 74 | 88% | 99% |
| Q8_0 | 1027 MB | 1052 MB | 130 | 88% | 96% |
| Q6_K | 845 MB | 870 MB | 151 | 87% | 95% |
| Q5_K_M | 796 MB | 821 MB | 156 | 87% | 93% |
| Q4_K_M | 750 MB | 775 MB | 167 | 87% | 88% |
| Q3_K_M | 687 MB | 712 MB | 156 | 70% | 70% |
| Q2_K | 646 MB | 671 MB | 171 | 4% | 2% |

8-bit costs nothing; quality holds to 4-bit, although 4-bit already loses two rows on the trained
patterns; below 4-bit it collapses. The Ollama model `tf2torch` uses Q6_K. Every file includes the same
174 MB vision part. In Ollama even F16 scores 88%, against 90% for the adapter in Transformers (one row differs).

## Limits

- The training pairs are generated from templates, so "seen patterns" measures how well the model learned
  those rules, not general skill.
- On TensorFlow functions it never saw, the model often just renames `tf.` to `torch.`
  (for example `torch.range`, `torch.Session`), which is wrong.
- It handles short snippets (the training window is 512 tokens), not whole files.

## Files

- `data/` - the dataset. Not stored in this repository; run `build_dataset.py` to create it.
- `outputs/lora/` - the trained LoRA adapter (base model: `unsloth/Qwen3.5-0.8B`)
- `outputs/comparison.json` - the scores of both models, per test row and per group (version 2). The models' answers (`outputs/answers_*.json`) and the test rows (`data/`) are not stored in this repository; `evaluate.py generate` recreates the answers
- `outputs/train_log.json` - loss history (version 2)
- `outputs/comparison_v1.json`, `train_log_v1.json` - the same for version 1
- `outputs/Modelfile` - the Ollama model definition
