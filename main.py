#!/usr/bin/env python3
"""Flow Model entry point.

    python main.py phases
    python main.py config show --section risk
    python main.py splits
"""

from __future__ import annotations

import sys

from flow_model.cli import main

if __name__ == "__main__":
    sys.exit(main())
