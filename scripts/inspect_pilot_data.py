import json
from collections import Counter

path = "results/teacher_targets_pilot.jsonl"

records = []
with open(path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

intent_counts = Counter()
slot_counts = Counter()

for record in records:
    target = json.loads(record["teacher_raw"])
    intent_counts[target["intent"]] += 1

    for field in ("date", "time", "place"):
        slot_counts[field] += target[field] is not None

print("TOTAL TRAINING EXAMPLES:", len(records))

print("\nEXAMPLES PER INTENT:")
for intent, count in sorted(intent_counts.items()):
    print(f"{intent:25} {count}")

print("\nNON-NULL SLOT COUNTS:")
for field, count in slot_counts.items():
    print(f"{field:10} {count}")

print("\nEXAMPLES WITH ALL SLOTS NULL:",
      sum(
          all(json.loads(r["teacher_raw"])[k] is None
              for k in ("date", "time", "place"))
          for r in records
      ))
