"""
api.py

FastAPI bridge between the Axiom frontend and the Zenith
behavioral intelligence engine. Provides entry submission
and insight retrieval via a lightweight REST interface.
"""

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from zenith.engine import run_zenith

app = FastAPI()

ENTRIES_FILE = Path("entries.json")


class Entry(BaseModel):
    date: str
    tasks_completed: int
    tasks_total: int
    deep_work_hours: float
    sleep_hours: float
    mood: int
    stress: int
    recovery: int


def load_entries() -> list:
    """Read entries from entries.json and return as list."""
    if not ENTRIES_FILE.exists():
        return []
    with open(ENTRIES_FILE, "r") as f:
        return json.load(f)


def save_entries(entries: list) -> None:
    """Write entries list to entries.json."""
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
    return {"status": "entry stored"}


@app.get("/insights")
def insights():
    entries = load_entries()
    if not entries:
        raise HTTPException(status_code=400, detail="No entries found")
    try:
        insights = run_zenith(entries)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return insights