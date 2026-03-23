from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json
from zenith.engine import run_zenith

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENTRIES_FILE = "entries.json"


class Entry(BaseModel):
    date: str
    tasks_completed: int
    tasks_total: int
    deep_work_hours: float
    sleep_hours: float
    mood: int
    stress: int
    recovery: int


def load_entries():
    with open(ENTRIES_FILE, "r") as f:
        return json.load(f)


def save_entries(entries):
    with open(ENTRIES_FILE, "w") as f:
        json.dump(entries, f, indent=2)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/submit-entry")
def submit_entry(entry: Entry):
    entries = load_entries()
    entries.append(entry.model_dump())
    save_entries(entries)
    return {"message": "Entry submitted."}


@app.get("/insights")
def insights():
    entries = load_entries()
    if not entries:
        raise HTTPException(status_code=400, detail="No entries found.")
    try:
        insights = run_zenith(entries)
        return insights
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))