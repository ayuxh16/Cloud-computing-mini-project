import json
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

INTENTS = (
    "alarm_set, alarm_query, calendar_set, calendar_query, "
    "calendar_remove, weather_query, datetime_query, recommendation_events"
)

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/student-pilot")
    parser.add_argument("--data", default="data/validation.jsonl")
    parser.add_argument("--output", default="results/pilot_predictions.jsonl")
    args = parser.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(4)

    print("Loading student model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(args.model)
    model.to(device)
    model.eval()

    predictions = []

    with open(args.data, encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Validation records: {len(records)}")

    for index, item in enumerate(records, 1):
        prompt = PROMPT_TEMPLATE.format(
            intents=INTENTS,
            command=item["text"].strip()
        )

        encoded = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False
        )

        input_ids = encoded["input_ids"][:, -384:].to(device)
        attention_mask = torch.ones_like(input_ids)

        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=100,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )

        new_tokens = generated[0, input_ids.shape[1]:]
        raw_prediction = tokenizer.decode(
            new_tokens,
            skip_special_tokens=True
        ).strip()

        predictions.append({
            "id": item["id"],
            "prediction": raw_prediction
        })

        if index % 10 == 0:
            print(f"Predicted {index}/{len(records)}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, "w", encoding="utf-8", newline="\n") as f:
        for item in predictions:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\nPredictions saved to {args.output}")
    print("\nFirst 3 predictions:")

    for item in predictions[:3]:
        print(json.dumps(item, ensure_ascii=False))

if __name__ == "__main__":
    main()
