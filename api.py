"""Thin FastAPI wrapper for Zenith v1.2 — persistence layer only."""

import json
import pathlib

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from zenith import analyze_data

app = FastAPI(title="Zenith API", version="1.2.0")

# ✅ CORS FIX (must be right after app creation)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ENTRIES_FILE = pathlib.Path("entries.json")


# ── Schema ───────────────────────────────────────────────────────────────────

class Entry(BaseModel):
    date: str
    tasks_completed: int
    tasks_total: int
    deep_work_hours: float
    sleep_hours: float
    mood: int
    stress: int
    recovery: int


# ── Helpers ──────────────────────────────────────────────────────────────────

def _read_entries() -> list[dict]:
    if not ENTRIES_FILE.exists():
        return []
    with open(ENTRIES_FILE, "r") as f:
        return json.load(f)


def _write_entries(entries: list[dict]) -> None:
    with open(ENTRIES_FILE, "w") as f:
        json.dump(entries, f, indent=2, default=str)


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "OK"}


@app.post("/submit-entry", status_code=201)
def submit_entry(entry: Entry):
    entries = _read_entries()
    entries.append(entry.model_dump())
    _write_entries(entries)
    return {"message": "Entry saved", "total_entries": len(entries)}


@app.get("/insights")
def insights():
    entries = _read_entries()

    if not entries:
        raise HTTPException(status_code=400, detail="No entries found")

    try:
        result = analyze_data(entries)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    return result