from __future__ import annotations

import argparse
import math
import random
from collections.abc import Sequence
from decimal import Decimal
from typing import Any


def removal_ratio(value: str | float) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "removal ratio must be a number between 0 and 1"
        ) from error
    if not math.isfinite(ratio) or not 0.0 <= ratio <= 1.0:
        raise argparse.ArgumentTypeError(
            "removal ratio must be a number between 0 and 1"
        )
    return ratio


def positive_int(value: str | int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError(
            "value must be a positive integer"
        ) from error
    if isinstance(value, float) or parsed <= 0 or str(value).strip() != str(parsed):
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def filter_generated_items(
    items: Sequence[dict[str, Any]],
    remove_ratio: str | float,
) -> list[dict[str, Any]]:
    ratio = removal_ratio(remove_ratio)
    ordered = sorted(items, key=lambda item: item["perplex"])
    keep_count = len(ordered) - math.floor(len(ordered) * ratio)
    return ordered[:keep_count]


def sample_generated_items(
    items: Sequence[dict[str, Any]],
    *,
    count: int,
    seed: int,
) -> list[dict[str, Any]]:
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError("sample count must be a positive integer")
    if not items:
        return []
    if count > len(items):
        raise ValueError(
            f"sample count {count} exceeds {len(items)} filtered candidates"
        )
    return random.Random(seed).sample(list(items), count)


def format_ratio_tag(value: str | float) -> str:
    ratio = removal_ratio(value)
    decimal = format(Decimal(str(ratio)).normalize(), "f")
    return "r" + decimal.replace(".", "p")
