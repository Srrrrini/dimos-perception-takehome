#!/usr/bin/env python3
"""Run the deterministic temporal semantic-map association evaluation."""

from __future__ import annotations

import json

from dimos.perception.semantic_map.evaluation import evaluate_controlled_temporal_tracking


def main() -> None:
    print(json.dumps(evaluate_controlled_temporal_tracking().as_dict(), sort_keys=True))


if __name__ == "__main__":
    main()
