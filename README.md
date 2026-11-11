# ArabicSense

**A benchmark for commonsense reasoning in Arabic**

[![Paper](https://img.shields.io/badge/WACL--4_2025-paper-b31b1b)](https://aclanthology.org/2025.wacl-1.1/)
[![Dataset](https://img.shields.io/badge/%F0%9F%A4%97_Dataset-ArabicSense-yellow)](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense)
[![Models](https://img.shields.io/badge/%F0%9F%A4%97_Collection-ArabicSense-orange)](https://huggingface.co/collections/Kamyar-zeinalipour/arabicsense-wacl-4-2025-6af47f5fabd9b1da26dbccd7)
[![Encoder experiments](https://img.shields.io/badge/GitHub-encoder_experiments-181717?logo=github)](https://github.com/EL-Amrany/Arabic-Commonsense-Reasoning)
[![License](https://img.shields.io/badge/code-Apache_2.0-blue)](LICENSE)

ArabicSense tests whether language models can tell sensible Arabic statements from nonsensical
ones and explain why. It accompanies the paper
[*ArabicSense: A Benchmark for Evaluating Commonsense Reasoning in Arabic with Large Language
Models*](https://aclanthology.org/2025.wacl-1.1/) (WACL-4 2025), the first benchmark of its kind for Arabic. This repository holds the
data and the fine-tuning and inference code for the causal LLMs (Gemma-2-9B, Llama-3.1-8B,
Mistral-7B). The encoder experiments (AraBERT, CAMeLBERT and others) are in
[EL-Amrany/Arabic-Commonsense-Reasoning](https://github.com/EL-Amrany/Arabic-Commonsense-Reasoning).

| Task | Input | Output | Train | Test |
|---|---|---|---:|---:|
| **A**: commonsense validation | two statements | which one does not make sense | 4,802 | 848 |
| **B**: explanation selection | a non-commonsense statement + three reasons | the reason that explains it | 4,802 | 848 |
| **C**: explanation generation | a non-commonsense statement | a reason in Arabic (three references) | 4,802 | 848 |

## The data

The three tasks are in [`data/`](data/) and on the Hugging Face Hub as
[`ArabicSense`](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense) (Task A), [`ArabicSense-task2`](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense-task2) (Task B)
and [`ArabicSense-task3`](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense-task3) (Task C).

```python
import pandas as pd

task_a = pd.read_json("data/task_a_test.jsonl.gz", lines=True)
print(task_a.iloc[0].to_dict())   # sentence1, sentence2, label (index of the nonsensical statement)
```

Statements were generated from Arabic Wikipedia articles with GPT-4 and checked by native Arabic
annotators; the data card in [`data/`](data/README.md) lists the fields.

## Fine-tuned models

| Base model | Adapter | Training data |
|---|---|---|
| [gemma-2-9b-it](https://huggingface.co/google/gemma-2-9b-it) | [gemma2-9b-CS-AR](https://huggingface.co/Kamyar-zeinalipour/gemma2-9b-CS-AR) | [ArabicCommonsense_DB_Gemma](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicCommonsense_DB_Gemma) |
| [Meta-Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B-Instruct) | [Llama3-8B-CS-AR](https://huggingface.co/Kamyar-zeinalipour/Llama3-8B-CS-AR) | [ArabicCommonsense_DB_Llama3](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicCommonsense_DB_Llama3) |
| [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3) | [Mistral-7b-CS-AR](https://huggingface.co/Kamyar-zeinalipour/Mistral-7b-CS-AR) | [ArabicCommonsense_DB_Mistral](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicCommonsense_DB_Mistral) |

Each adapter was trained on all three tasks at once; the `ArabicCommonsense_DB_*` datasets hold
the tasks in each model's chat format.

```bash
pip install -r requirements.txt
python inference/generate.py --model gemma2 --limit 20          # fine-tuned
python inference/generate.py --model gemma2 --base --limit 20   # zero-shot
```

## Results

Zero-shot → fine-tuned on the ArabicSense test set (Tables 5–7 of the paper). Tasks A and B: F1;
Task C: BERTScore F1 against the reference explanations.

| Model | Task A F1 | Task B F1 | Task C BERTScore F1 |
|---|---|---|---|
| gemma-2-9b-it | 0.867 → **0.947** | 0.921 → **0.944** | 0.656 → **0.759** |
| Llama-3.1-8B-Instruct | 0.659 → **0.945** | 0.863 → **0.930** | 0.747 → **0.773** |
| Mistral-7B-Instruct-v0.3 | 0.601 → **0.948** | 0.805 → **0.934** | 0.728 → **0.771** |

Accuracy, precision and recall for every setting are in
[`paper_results/causal_llms.csv`](paper_results/causal_llms.csv). Encoder results are in the paper.

## Training

```bash
scripts/train_gemma2.sh     # also train_llama3.sh, train_mistral.sh
```

LoRA (r = 16, α = 32, dropout 0.1) on all attention and MLP projections plus the embeddings and
LM head; AdamW, learning rate 1e-4 with cosine decay, weight decay 1e-4; 8 sequences per GPU ×
4 gradient-accumulation steps; up to 1,024 tokens; bf16, DeepSpeed, Flash Attention 2, seed 100.
Gemma-2 and Llama-3.1 were trained for one epoch, Mistral for three.

## Citation

```bibtex
@inproceedings{lamsiyah-etal-2025-arabicsense,
    title = "{A}rabic{S}ense: A Benchmark for Evaluating Commonsense Reasoning in {A}rabic with Large Language Models",
    author = "Lamsiyah, Salima  and
      Zeinalipour, Kamyar  and
      El amrany, Samir  and
      Brust, Matthias  and
      Maggini, Marco  and
      Bouvry, Pascal  and
      Schommer, Christoph",
    editor = "Ezzini, Saad  and
      Alami, Hamza  and
      Berrada, Ismail  and
      Benlahbib, Abdessamad  and
      El Mahdaouy, Abdelkader  and
      Lamsiyah, Salima  and
      Derrouz, Hatim  and
      Haddad Haddad, Amal  and
      Jarrar, Mustafa  and
      El-Haj, Mo  and
      Mitkov, Ruslan  and
      Rayson, Paul",
    booktitle = "Proceedings of the 4th Workshop on {A}rabic Corpus Linguistics ({WACL}-4)",
    month = jan,
    year = "2025",
    address = "Abu Dhabi, UAE",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.wacl-1.1/",
    pages = "1--11"
}
```

## License

Code: [Apache License 2.0](LICENSE). Data: CC BY-SA 4.0 (derived from Wikipedia). Adapters follow
the license of their base model.
