import json
from collections import defaultdict

path = "results/teacher_targets_pilot.jsonl"
groups = defaultdict(list)

with open(path, encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue
        record = json.loads(line)
        target = json.loads(record["teacher_raw"])
        groups[target["intent"]].append((record, target))

for intent in ("recommendation_events", "datetime_query", "alarm_query"):
    print(f"\n{'=' * 65}")
    print(f"{intent.upper()} — {len(groups[intent])} examples")
    print("=" * 65)

    for record, target in groups[intent][:15]:
        print(f"\n{record['id']}: {record['text']}")
        print("  date:", target["date"],
              "| time:", target["time"],
              "| place:", target["place"])

print("\n\n========== SLOT-BEARING EXAMPLES ==========")
shown = 0
for intent, examples in sorted(groups.items()):
    for record, target in examples:
        if any(target[k] is not None for k in ("date", "time", "place")):
            print(f"{record['id']} [{intent}]: {record['text']}")
            print(" ", {k: target[k] for k in ("date", "time", "place")})
            shown += 1
            if shown >= 20:
                break
    if shown >= 20:
        break
