import json
from collections import Counter

gold_path = "data/validation.jsonl"
pred_path = "results/pilot_predictions.jsonl"

def read_jsonl(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records

gold = {x["id"]: x for x in read_jsonl(gold_path)}
predictions = {
    x["id"]: x["prediction"]
    for x in read_jsonl(pred_path)
}

def normalize(value):
    return value.strip().lower() if isinstance(value, str) else value

intent_errors = Counter()
field_errors = Counter()
wrong_examples = []

for record_id, actual in gold.items():
    raw = predictions.get(record_id, "")

    try:
        predicted = json.loads(raw)
        if not isinstance(predicted, dict):
            raise ValueError("Not a JSON object")
    except Exception:
        field_errors["invalid_json"] += 1
        wrong_examples.append(
            (record_id, actual, raw, "INVALID JSON")
        )
        continue

    errors = []

    for field in ("intent", "date", "time", "place"):
        expected = actual[field]
        received = predicted.get(field)

        if field == "intent":
            correct = received == expected
            if not correct:
                intent_errors[(expected, str(received))] += 1
        else:
            correct = normalize(received) == normalize(expected)

        if not correct:
            field_errors[field] += 1
            errors.append(
                f"{field}: expected={expected!r}, predicted={received!r}"
            )

    if errors:
        wrong_examples.append(
            (record_id, actual, raw, "; ".join(errors))
        )

print("\n========== ERROR SUMMARY ==========")
for field, count in field_errors.items():
    print(f"{field}: {count}")

print("\n========== INTENT CONFUSIONS ==========")
for (expected, received), count in intent_errors.most_common(15):
    print(f"Expected {expected} -> Predicted {received}: {count}")

print("\n========== FIRST 20 INCORRECT EXAMPLES ==========")
for record_id, actual, raw, errors in wrong_examples[:20]:
    print(f"\nID: {record_id}")
    print(f"COMMAND: {actual['text']}")
    print(f"GOLD: {actual['intent']}, date={actual['date']}, "
          f"time={actual['time']}, place={actual['place']}")
    print(f"PREDICTION: {raw}")
    print(f"ERRORS: {errors}")

print(f"\nTotal incorrect examples: {len(wrong_examples)}")
print(f"Missing predictions: {len(set(gold) - set(predictions))}")
