"""Compatibility entry point for answering cases with GPT-4o-mini only."""

from __future__ import annotations

import sys

from answer_models import main


if __name__ == "__main__":
    if "--models" not in sys.argv:
        sys.argv.extend(["--models", "gpt-4o-mini"])
    main()