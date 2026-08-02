#!/usr/bin/env python3
"""Evaluate C2R with thresholds selected on the reported run itself."""

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from evaluation.c2r import evaluate_run_in_sample, write_report_atomic


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output = args.output or args.run / "c2r_evaluation_validation_tuned.json"
    report = evaluate_run_in_sample(args.run)
    write_report_atomic(output, report)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
