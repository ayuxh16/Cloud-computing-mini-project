import json
from collections import Counter

path = "data/commands_pool.jsonl"

records = []
with open(path, encoding="utf-8") as f:
    for line in f:
        if line.strip():
            records.append(json.loads(line))

print("TOTAL COMMANDS:", len(records))
print("\nFIRST 15 COMMANDS:")
for r in records[:15]:
    print(r["id"], ":", r["text"])

print("\nCOMMAND LENGTH DISTRIBUTION:")
lengths = [len(r["text"].split()) for r in records]
print("Average words:", round(sum(lengths) / len(lengths), 2))
print("Shortest:", min(lengths), "Longest:", max(lengths))

print("\nKEYWORD COVERAGE (rough indicators, not ground-truth labels):")
groups = {
    "alarm": ["alarm", "wake me", "wake up"],
    "calendar": ["calendar", "meeting", "event", "schedule"],
    "weather": ["weather", "forecast", "temperature"],
    "datetime": ["what time is it", "current time", "what date"],
    "recommendations": ["activity", "activities", "recommend", "events nearby"]
}

for group, keywords in groups.items():
    count = sum(
        any(k in r["text"].lower() for k in keywords)
        for r in records
    )
    print(f"{group:20}: {count}")

