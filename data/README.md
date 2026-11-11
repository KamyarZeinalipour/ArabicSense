# ArabicSense data

| File | Task | Fields |
|---|---|---|
| `task_a_{train,test}.jsonl.gz` | A: commonsense validation | `sentence1`, `sentence2`, `label` (index of the statement that does not make sense) |
| `task_b_{train,test}.jsonl.gz` | B: explanation selection | `non-commonsense statement`, `sentence1`–`sentence3` (candidate reasons), `label` (index of the correct reason) |
| `task_c_{train,test}.jsonl.gz` | C: explanation generation | `non-commonsense statement`, `reason1`–`reason3` (reference explanations) |

Each task has 4,802 training and 848 test items, all in Modern Standard Arabic and built from the
same set of Arabic Wikipedia articles. The same data is on the Hugging Face Hub:
[`ArabicSense`](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense),
[`ArabicSense-task2`](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense-task2),
[`ArabicSense-task3`](https://huggingface.co/datasets/Kamyar-zeinalipour/ArabicSense-task3).

License: CC BY-SA 4.0 (derived from Wikipedia).
