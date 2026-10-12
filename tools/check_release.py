"""Check curated release files for missing links, large artifacts and local source paths."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote

ROOT = Path(__file__).resolve().parents[1]
ABSOLUTE_PATH = re.compile(r"(?<![A-Za-z0-9])(?:[A-Za-z]:[\\/]|/(?:home|Users|mnt|tmp)/)")
TEXT_SUFFIXES = {".py", ".md", ".json", ".jsonl", ".yaml", ".yml", ".toml", ".csv", ".txt", ".cmd", ".ini"}
PUBLIC = [
    "README.md",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    ".gitignore",
    ".gitattributes",
    "启动跟踪GUI.cmd",
    "src",
    "research",
    "configs",
    "tests",
    "tools",
    "results",
    "protocols",
    "docs",
    "provenance",
    "manifests",
    ".github",
]


def candidates():
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
        )
        return [
            ROOT / p
            for p in result.stdout.decode("utf-8").split("\0")
            if p and (ROOT / p).is_file()
        ]
    result = []
    for name in PUBLIC:
        path = ROOT / name
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(
                p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts
            )
    return result


def check():
    errors = []
    files = candidates()
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if path.stat().st_size > 10 * 1024 * 1024 or path.suffix.lower() in {
            ".pt",
            ".onnx",
            ".mp4",
            ".npz",
            ".zip",
        }:
            errors.append(f"Large/research artifact in release: {rel}")
        if path.suffix.lower() in TEXT_SUFFIXES:
            for line_number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
                if ABSOLUTE_PATH.search(line):
                    errors.append(f"Machine-specific absolute path: {rel}:{line_number}")
        if path.suffix == ".py" and path.parts[len(ROOT.parts)] in {
            "src",
            "tools",
            "tests",
            "research",
        }:
            text = path.read_text(encoding="utf-8-sig")
            if re.search(r"(?i)E:[\\/]+CV|C:[\\/]+Users[\\/]+LG|reports[/\\]a2_", text):
                errors.append(f"Draft path dependency in code: {rel}")
        if path.suffix == ".md":
            for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8-sig")):
                link = link.strip().split(' "')[0].strip("<>").split("#")[0]
                if not link or re.match(r"^[a-z][a-z0-9+.-]*:", link, re.IGNORECASE):
                    continue
                if not (path.parent / unquote(link)).exists():
                    errors.append(f"Broken link: {rel} -> {link}")
    report = {
        "passed": not errors,
        "files": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "errors": errors,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    check()
