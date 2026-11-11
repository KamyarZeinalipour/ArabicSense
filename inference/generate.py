"""Answer the ArabicSense test prompts with a released LoRA adapter (or the base model).

    python inference/generate.py --model mistral            # fine-tuned adapter
    python inference/generate.py --model mistral --base     # zero-shot base model

The prompts are the test split of `Kamyar-zeinalipour/ArabicCommonsense_DB_<Model>`, which holds
the three ArabicSense tasks in each model's chat format. Each row is split at the model's
assistant marker into prompt and gold answer. Decoding follows the paper's runs
(temperature 0.1, top-p 0.95, top-k 50, repetition penalty 1.1, up to 512 new tokens).
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from datasets import load_dataset
from peft import PeftModel
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "mistral": ("mistralai/Mistral-7B-Instruct-v0.3", "Kamyar-zeinalipour/Mistral-7b-CS-AR", "Mistral", "[/INST]"),
    "llama3": ("meta-llama/Meta-Llama-3.1-8B-Instruct", "Kamyar-zeinalipour/Llama3-8B-CS-AR", "Llama3",
               "<|start_header_id|>assistant<|end_header_id|>\n\n"),
    "gemma2": ("google/gemma-2-9b-it", "Kamyar-zeinalipour/gemma2-9b-CS-AR", "Gemma", "<start_of_turn>model\n"),
}
END_TOKENS = ["</s>", "<|end_of_text|>", "<|eot_id|>", "<end_of_turn>", "<eos>"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--base", action="store_true", help="Use the base model without the adapter.")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    base_id, adapter_id, db, marker = MODELS[args.model]
    data = load_dataset(f"Kamyar-zeinalipour/ArabicCommonsense_DB_{db}", split="test")
    rows = data.select(range(args.limit)) if args.limit else data

    tokenizer = AutoTokenizer.from_pretrained(base_id)
    model = AutoModelForCausalLM.from_pretrained(base_id, torch_dtype=torch.bfloat16, device_map="auto")
    if not args.base:
        model = PeftModel.from_pretrained(model, adapter_id).merge_and_unload()
    model.eval()

    out = []
    for text in tqdm(rows["text"], desc=args.model):
        head, gold = text.split(marker, 1)
        prompt = head + marker
        for t in END_TOKENS:
            gold = gold.replace(t, "")
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=512, do_sample=True, temperature=0.1, top_k=50,
                                 top_p=0.95, repetition_penalty=1.1, eos_token_id=tokenizer.eos_token_id,
                                 pad_token_id=tokenizer.eos_token_id)
        reply = tokenizer.decode(ids[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        out.append({"prompt": prompt, "gold": gold.strip(), "output": reply})

    path = Path(args.output or f"results/{args.model}_{'base' if args.base else 'finetuned'}.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(out).to_csv(path, index=False)
    print(f"Saved {len(out)} answers to {path}")


if __name__ == "__main__":
    main()
