#!/usr/bin/env python3
"""One-off helper to create the database schema without starting the server.

Usage:
    python scripts/init_db.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import init_db  # noqa: E402

if __name__ == "__main__":
    init_db()
    print("database initialized")
