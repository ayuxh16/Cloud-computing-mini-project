import json
from pathlib import Path
import requests

MODEL = "phi3.5"
OLLAMA_URL = "http://localhost:11434/api/generate"

prompt_template = Path("prompts/teacher_prompt.txt").read_text(encoding="utf-8")
with open("data/commands_pool.jsonl", encoding="utf-8") as f:
    record = json.loads(next(f))

prompt = prompt_template + "\n" + record["text"]

response = requests.post(
    OLLAMA_URL,
    json={
        "model": MODEL,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": 0,
            "seed": 42,
            "num_predict": 128
        }
    },
    timeout=300
)
response.raise_for_status()

raw = response.json()["response"]

print("COMMAND:", record["text"])
print("TEACHER RAW RESPONSE:")
print(raw)

try:
    parsed = json.loads(raw)
    expected = {"intent", "date", "time", "place"}
    print("EXACT FOUR KEYS:", set(parsed.keys()) == expected)
    print("JSON VALID:", True)
except (json.JSONDecodeError, AttributeError):
    print("JSON VALID:", False)
