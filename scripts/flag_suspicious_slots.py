import json
from collections import Counter

path = "results/teacher_targets_pilot.jsonl"

suspicious = Counter()
examples = []

with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        record = json.loads(line)
        target = json.loads(record["teacher_raw"])
        command = record["text"].lower()

        for field in ("date", "time", "place"):
            value = target[field]

            if value is None:
                continue

            value_lower = value.lower().strip()
            reasons = []

            if value_lower in {
                "time", "date", "place", "city",
                "location", "day", "hour"
            }:
                reasons.append("generic_slot_word")

            if field == "time" and value_lower in {
                "morning", "afternoon", "evening", "night"
            }:
                pass

            if value_lower not in command:
                reasons.append("not_exact_substring")

            if reasons:
                suspicious.update(reasons)
                examples.append(
                    (record["id"], command, field, value, reasons)
                )

print("SUSPICIOUS OUTPUT COUNTS:")
for reason, count in suspicious.items():
    print(f"{reason}: {count}")

print("\nEXAMPLES:")
for item in examples[:30]:
    print(item)

print("\nTOTAL FLAGGED SLOT VALUES:", len(examples))
