# Fine-tuning Qwen3.5-0.8B to translate TensorFlow 1.x code to PyTorch

A small LoRA fine-tune done on a laptop GPU (RTX 3050 Ti, 4 GB) with [Unsloth](https://github.com/unslothai/unsloth).
The model learns to rewrite TensorFlow 1.x code (placeholders, `tf.Variable`, `tf.Session`, `feed_dict`,
`AdamOptimizer`...) as PyTorch.

## Steps

| Step | Script | What it does |
|---|---|---|
| 1 | `build_dataset.py` | Builds TensorFlow -> PyTorch pairs: 798 train, 88 validation, 94 test. Every reference answer is executed to prove it runs. |
| 2 | `train.py` | LoRA fine-tune: 16-bit, rank 16, batch size 1, gradient accumulation 4, max sequence length 512, 3 epochs (about 7 minutes). |
| 3 | `evaluate.py` | Runs the original and the fine-tuned model on the test set and scores both. |
| 4 | `export_ollama.py` | Merges the LoRA add-on, quantises to q4_K_M and creates the Ollama model `tf2torch`. |

`exec_check.py` runs generated code in a separate CPU process and compares its numbers with the reference.

## Results on the test set (never used in training)

"Runs correctly" means the answer was executed and produced the same numbers as the reference PyTorch code.

| Test group | Model | Valid Python | No TensorFlow left | Runs correctly | Exact match |
|---|---|---|---|---|---|
| Seen patterns (65) | original | 95% | 40% | 4% of 50 | 0% |
| Seen patterns (65) | fine-tuned | 100% | 100% | 100% of 50 | 100% |
| Unseen patterns (21) | original | 95% | 62% | 10% of 21 | 0% |
| Unseen patterns (21) | fine-tuned | 100% | 100% | 48% of 21 | 43% |
| Real project code (8) | original | 100% | 38% | 0% of 6 | 0% |
| Real project code (8) | fine-tuned | 100% | 100% | 67% of 6 | 12% |
| All (94) | original | 96% | 45% | 5% of 77 | 0% |
| All (94) | fine-tuned | 100% | 100% | 83% of 77 | 80% |

Validation loss: 0.0016 -> 0.0011 -> 0.0004 over the three epochs.

## Limits

- The training pairs are generated from templates, so "seen patterns" measures how well the model learned
  those rules, not general skill.
- On TensorFlow functions it never saw, the model often just renames `tf.` to `torch.`
  (for example `torch.range`, `torch.Session`), which is wrong.
- It handles short snippets (the training window is 512 tokens), not whole files.

## Files

- `data/` - the dataset
- `outputs/lora/` - the trained LoRA adapter (base model: `unsloth/Qwen3.5-0.8B`)
- `outputs/comparison.json` - every test answer from both models, with scores
- `outputs/train_log.json` - loss history
- `outputs/Modelfile` - the Ollama model definition
