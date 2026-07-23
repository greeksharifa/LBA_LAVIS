from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCANNER_PATH = Path(__file__).resolve()

EXCLUDED_PARTS = {
    ".git",
    ".cache",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "artifacts",
    "build",
    "cache",
    "caches",
    "checkpoint",
    "checkpoints",
    "data",
    "dataset",
    "datasets",
    "dist",
    "generated",
    "node_modules",
    "output",
    "outputs",
    "runs",
    "venv",
    "wandb",
}
EXCLUDED_SUFFIXES = {
    ".7z",
    ".avi",
    ".bin",
    ".bmp",
    ".bz2",
    ".ckpt",
    ".gif",
    ".gz",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".mov",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".pdf",
    ".png",
    ".pt",
    ".pth",
    ".tar",
    ".tgz",
    ".webm",
    ".webp",
    ".xz",
    ".zip",
}
FORBIDDEN_FILENAMES = (
    re.compile(r"^\.env(?:\..+)?$", re.IGNORECASE),
    re.compile(r"^cookies?(?:\.(?:txt|json))?$", re.IGNORECASE),
)
FORBIDDEN_CONTENT = {
    "personal home path": re.compile(rb"/home/[^/\s\"']+/"),
    "numbered data mount": re.compile(rb"/data\d+/"),
    "NAS mount": re.compile(rb"/nas-[^/\s\"']+/"),
    "ipdb residue": re.compile(rb"\bipdb\b"),
    "pdb breakpoint": re.compile(rb"\bpdb\.set_trace\s*\("),
    "built-in breakpoint": re.compile(rb"\bbreakpoint\s*\("),
    "fixed CUDA devices": re.compile(
        rb"\bCUDA_VISIBLE_DEVICES(?:[\"']?\])?\s*(?:=|:)\s*"
        rb"[\"']?\d+(?:\s*,\s*\d+)*[\"']?"
    ),
    "AWS access key": re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    "OpenAI-style key": re.compile(rb"\bsk[-_][A-Za-z0-9_-]{8,}\b"),
    "GitHub token": re.compile(rb"\bghp_[A-Za-z0-9]{8,}\b"),
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential assignment": re.compile(
        rb"(?i)\b(?:api[_ -]?key|access[_ -]?token|secret[_ -]?key|password)"
        rb"[\"']?\s*(?::=|=|:)\s*[\"']?[^\s\"'${}][^\s\"']*"
    ),
    "session assignment": re.compile(
        rb"(?i)\b(?:cookie|session_id|session_token)"
        rb"[\"']?\s*(?::=|=|:)\s*[\"']?[^\s\"'${}][^\s\"']*"
    ),
}


def repository_candidates() -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
    )
    return [
        REPOSITORY_ROOT / raw_path.decode("utf-8")
        for raw_path in result.stdout.split(b"\0")
        if raw_path
    ]


def is_excluded(path: Path) -> bool:
    relative_path = path.relative_to(REPOSITORY_ROOT)
    return (
        path == SCANNER_PATH
        or bool(EXCLUDED_PARTS.intersection(relative_path.parts))
        or path.suffix.lower() in EXCLUDED_SUFFIXES
    )


def test_repository_contains_no_sensitive_filenames_or_content() -> None:
    violations: list[str] = []

    for path in repository_candidates():
        if is_excluded(path) or not path.is_file():
            continue

        relative_path = path.relative_to(REPOSITORY_ROOT)
        if any(pattern.fullmatch(path.name) for pattern in FORBIDDEN_FILENAMES):
            violations.append(f"{relative_path}: forbidden filename")
            continue

        content = path.read_bytes()
        if b"\0" in content:
            continue

        for label, pattern in FORBIDDEN_CONTENT.items():
            for match in pattern.finditer(content):
                line_number = content.count(b"\n", 0, match.start()) + 1
                violations.append(f"{relative_path}:{line_number}: {label}")

    assert not violations, "Repository sanitization violations:\n" + "\n".join(violations)
