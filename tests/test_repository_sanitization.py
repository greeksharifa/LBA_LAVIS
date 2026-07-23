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
    "cache",
    "caches",
    "node_modules",
    "venv",
}
EXCLUDED_SUFFIXES = {
    ".7z",
    ".avi",
    ".arrow",
    ".bin",
    ".bmp",
    ".bz2",
    ".ckpt",
    ".db",
    ".feather",
    ".gif",
    ".gz",
    ".h5",
    ".hdf5",
    ".ico",
    ".jpeg",
    ".jpg",
    ".mkv",
    ".lmdb",
    ".mov",
    ".mp3",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".parquet",
    ".pdf",
    ".pickle",
    ".pkl",
    ".png",
    ".pt",
    ".pth",
    ".safetensors",
    ".sqlite",
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
    "AWS access key": re.compile(rb"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    "OpenAI-style key": re.compile(rb"\bsk[-_][A-Za-z0-9_-]{8,}\b"),
    "GitHub token": re.compile(
        rb"\b(?:(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{8,}|"
        rb"github_pat_[A-Za-z0-9_]{20,})\b"
    ),
    "private key": re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "credential assignment": re.compile(
        rb"(?i)\b(?:[a-z0-9]+_)*(?:api_key|secret_access_key|secret_key|"
        rb"access_token|github_token|password|session_token|session_id|cookie)"
        rb"[\"']?\s*(?::=|=|:)\s*"
        rb"(?!(?:os\.)?(?:environ(?:\.get)?\s*(?:\[|\()|getenv\s*\()|\$)"
        rb"(?:[rubf]{0,2}[\"'][^\"'\n]+[\"']|"
        rb"(?!(?:None|True|False)\b)[A-Za-z0-9][A-Za-z0-9._/-]{3,})"
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


def is_excluded(
    path: Path,
    *,
    repository_root: Path = REPOSITORY_ROOT,
    scanner_path: Path | None = SCANNER_PATH,
) -> bool:
    relative_path = path.relative_to(repository_root)
    return (
        (scanner_path is not None and path == scanner_path)
        or bool(EXCLUDED_PARTS.intersection(relative_path.parts))
        or path.suffix.lower() in EXCLUDED_SUFFIXES
    )


def scan_paths(
    paths: list[Path],
    *,
    repository_root: Path = REPOSITORY_ROOT,
    scanner_path: Path | None = SCANNER_PATH,
) -> list[str]:
    violations: list[str] = []

    for path in paths:
        if (
            is_excluded(
                path,
                repository_root=repository_root,
                scanner_path=scanner_path,
            )
            or not path.is_file()
        ):
            continue

        relative_path = path.relative_to(repository_root)
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

    return violations


def test_repository_contains_no_sensitive_filenames_or_content() -> None:
    violations = scan_paths(repository_candidates())

    assert not violations, "Repository sanitization violations:\n" + "\n".join(violations)


def test_scanner_checks_code_and_docs_under_data_like_directories(tmp_path) -> None:
    generated_config = tmp_path / "generated/config.py"
    generated_config.parent.mkdir(parents=True)
    generated_config.write_text('OPENAI_API_KEY = "literal-fixture-value"\n')
    data_readme = tmp_path / "data/README.md"
    data_readme.parent.mkdir(parents=True)
    data_readme.write_text("server root: /home/example/private\n")

    violations = scan_paths(
        [generated_config, data_readme],
        repository_root=tmp_path,
        scanner_path=None,
    )

    assert any("generated/config.py:1: credential assignment" in item for item in violations)
    assert any("data/README.md:1: personal home path" in item for item in violations)


def test_scanner_detects_prefixed_credentials_and_github_tokens(tmp_path) -> None:
    fixture = tmp_path / "config.py"
    fixture.write_text(
        "\n".join(
            [
                'OPENAI_API_KEY = "literal-api-value"',
                'AWS_SECRET_ACCESS_KEY = "literal-aws-value"',
                'GITHUB_TOKEN = "gho_fixturetoken123"',
                'DATABASE_PASSWORD = "literal-password-value"',
                'APP_SESSION_TOKEN = "literal-session-value"',
                'SECOND_TOKEN = "ghu_fixturetoken123"',
                'THIRD_TOKEN = "ghs_fixturetoken123"',
                'FOURTH_TOKEN = "ghr_fixturetoken123"',
            ]
        )
    )

    violations = scan_paths(
        [fixture],
        repository_root=tmp_path,
        scanner_path=None,
    )

    assert sum("credential assignment" in item for item in violations) >= 5
    assert sum("GitHub token" in item for item in violations) == 4


def test_scanner_allows_environment_credential_references(tmp_path) -> None:
    fixture = tmp_path / "config.py"
    fixture.write_text(
        "\n".join(
            [
                'OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]',
                'AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")',
                'GITHUB_TOKEN = getenv("GITHUB_TOKEN")',
                'DATABASE_PASSWORD = os.environ.get("DATABASE_PASSWORD")',
                'APP_SESSION_TOKEN = os.getenv("APP_SESSION_TOKEN")',
            ]
        )
    )

    violations = scan_paths(
        [fixture],
        repository_root=tmp_path,
        scanner_path=None,
    )

    assert violations == []


def test_scanner_detects_bare_modern_credentials_in_structured_text(
    tmp_path,
) -> None:
    json_fixture = tmp_path / "config.json"
    json_fixture.write_text(
        '{"value": "github_pat_fixtureToken1234567890"}\n'
    )
    yaml_fixture = tmp_path / "config.yaml"
    yaml_fixture.write_text("value: ASIAABCDEFGHIJKLMNOP\n")
    url_fixture = tmp_path / "url.txt"
    url_fixture.write_text(
        "https://example.invalid/github_pat_secondFixture1234567890/status\n"
    )
    docs_fixture = tmp_path / "README.md"
    docs_fixture.write_text("temporary identifier: ASIA1234567890ABCDEF\n")

    violations = scan_paths(
        [json_fixture, yaml_fixture, url_fixture, docs_fixture],
        repository_root=tmp_path,
        scanner_path=None,
    )

    assert sum("GitHub token" in item for item in violations) == 2
    assert sum("AWS access key" in item for item in violations) == 2
