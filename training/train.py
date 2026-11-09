import os
import json
import random
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "EleutherAI/pythia-70m"

INTENTS = (
    "alarm_set, alarm_query, calendar_set, calendar_query, "
    "calendar_remove, weather_query, datetime_query, recommendation_events"
)

VALID_INTENTS = set(INTENTS.split(", "))
EXPECTED_KEYS = {"intent", "date", "time", "place"}

PROMPT_TEMPLATE = """You are a command-parsing language model.
Extract the intent and slots from the user's command.

Return exactly one JSON object with exactly these four keys:
"intent", "date", "time", "place".

Allowed intents: {intents}

Rules:
- Copy slot values exactly as they appear in the command.
- Use the shortest meaningful phrase.
- Exclude prepositions such as at, on, in and for.
- Do not convert dates or times.
- Use null for slots not mentioned.
- Never invent information.
- Return JSON only, without explanations or markdown.

Command: {command}
JSON:
"""


def load_records(path):
    records = []
    required = {"id", "text", "teacher_raw"}

    with open(path, encoding="utf-8") as f:
        for line_number, line in enumerate(f, 1):
            if not line.strip():
                continue

            item = json.loads(line)

            if not isinstance(item, dict) or set(item.keys()) != required:
                raise ValueError(
                    f"Invalid teacher target at line {line_number}"
                )

            if not isinstance(item["teacher_raw"], str):
                raise ValueError(
                    f"Invalid teacher response at line {line_number}"
                )

            target = json.loads(item["teacher_raw"])

            if (
                not isinstance(target, dict)
                or set(target.keys()) != EXPECTED_KEYS
                or not isinstance(target.get("intent"), str)
                or target["intent"] not in VALID_INTENTS
            ):
                raise ValueError(
                    f"Invalid teacher JSON at line {line_number}"
                )

            records.append(item)

    if not records:
        raise RuntimeError("Training dataset is empty.")

    return records


class DistillationDataset(Dataset):
    def __init__(self, records, tokenizer, max_length):
        self.examples = []

        for item in records:
            prompt = PROMPT_TEMPLATE.format(
                intents=INTENTS,
                command=item["text"].strip(),
            )

            # Exact original teacher-generated response.
            target = item["teacher_raw"]

            prompt_ids = tokenizer(
                prompt, add_special_tokens=False
            )["input_ids"]

            target_ids = tokenizer(
                target, add_special_tokens=False
            )["input_ids"]

            target_ids = target_ids + [tokenizer.eos_token_id]

            if len(target_ids) >= max_length:
                raise ValueError(
                    f"Target too long for record {item['id']}"
                )

            available_prompt = max_length - len(target_ids)
            prompt_ids = prompt_ids[-available_prompt:]

            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

            self.examples.append({
                "input_ids": input_ids,
                "labels": labels,
            })

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


class DistillationCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        max_len = max(len(x["input_ids"]) for x in batch)

        input_ids = []
        attention_mask = []
        labels = []

        for item in batch:
            padding = max_len - len(item["input_ids"])

            input_ids.append(
                item["input_ids"] + [self.pad_token_id] * padding
            )
            attention_mask.append(
                [1] * len(item["input_ids"]) + [0] * padding
            )
            labels.append(item["labels"] + [-100] * padding)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default="results/teacher_targets.jsonl",
    )
    parser.add_argument("--output", default="models/student")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive.")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(min(4, os.cpu_count() or 1))

    device = torch.device("cpu")

    print("Loading original teacher responses...", flush=True)
    records = load_records(args.data)

    print(f"Training examples: {len(records)}", flush=True)
    print("Validation data is not loaded or used.", flush=True)
    print(f"Loading base model: {MODEL_NAME}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.eos_token_id is None:
        raise RuntimeError("Tokenizer has no EOS token.")

    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
    )

    model.to(device)
    model.train()

    dataset = DistillationDataset(
        records, tokenizer, args.max_length
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=DistillationCollator(tokenizer.pad_token_id),
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()

        total_loss = 0.0
        batch_count = 0

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} started",
            flush=True,
        )

        for step, batch in enumerate(loader):
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(**batch)
            loss = outputs.loss

            total_loss += loss.item()
            batch_count += 1

            (loss / args.gradient_accumulation).backward()

            if (
                (step + 1) % args.gradient_accumulation == 0
                or step + 1 == len(loader)
            ):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 1.0
                )
                optimizer.step()
                optimizer.zero_grad()

            if (step + 1) % 100 == 0:
                print(
                    f"Epoch {epoch + 1}: "
                    f"{step + 1}/{len(loader)} batches, "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        average_loss = total_loss / max(batch_count, 1)

        print(
            f"Epoch {epoch + 1} average loss: {average_loss:.4f}",
            flush=True,
        )

        checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch + 1}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        model.save_pretrained(checkpoint_dir)
        tokenizer.save_pretrained(checkpoint_dir)

    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\nTRAINING COMPLETE", flush=True)
    print(f"Student model saved to: {output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
