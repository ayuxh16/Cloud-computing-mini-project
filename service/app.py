import os
import socket
import threading
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_DIR = os.getenv("MODEL_DIR", "/models/student")
MODEL_VERSION = os.getenv("MODEL_VERSION", "pythia-70m-student")
MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "96"))

tokenizer = None
model = None
generation_lock = threading.Lock()

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


@asynccontextmanager
async def lifespan(app: FastAPI):
    global tokenizer, model

    if not os.path.isdir(MODEL_DIR):
        raise RuntimeError(f"Student model directory not found: {MODEL_DIR}")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, local_files_only=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        torch_dtype=torch.float32
    )
    model.to("cpu")
    model.eval()

    yield

    del model
    del tokenizer


app = FastAPI(
    title="Lightweight Command Parser",
    version="1.0.0",
    lifespan=lifespan
)


class ParseRequest(BaseModel):
    text: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    if model is None:
        raise HTTPException(status_code=503, detail="Student model not loaded")
    return {"status": "healthy", "model": MODEL_VERSION}


@app.post("/parse")
def parse_command(payload: ParseRequest, response: Response):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Student model unavailable")

    prompt = PROMPT_TEMPLATE.format(
        intents=INTENTS,
        command=payload.text.strip()
    )

    try:
        with generation_lock:
            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                    num_beams=1,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            generated = output[0][inputs["input_ids"].shape[1]:]
            prediction = tokenizer.decode(
                generated, skip_special_tokens=True
            ).strip()

        response.headers["X-Served-By"] = socket.gethostname()
        response.headers["X-Model-Version"] = MODEL_VERSION

        # Keep the raw output so the evaluation pipeline can detect
        # invalid JSON and malformed predictions.
        return {"prediction": prediction}

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}"
        )
