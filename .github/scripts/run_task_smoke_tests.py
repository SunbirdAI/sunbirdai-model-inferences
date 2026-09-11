#!/usr/bin/env python3
"""Run each fixture below through src/handler.py and fail if any task comes
back as {"Error": ...} instead of a real result.

The previous CI check ran the handler with a task-less input, which only
exercises the "Task is missing" guard clause in handler.handler() and can
never catch a broken task — that's how a tokenizer regression in
auto_detect_language/language_classify shipped to production undetected.

Limited to tasks backed by a purely local model: translate/transcribe/tts
fixtures need GCP credentials and a storage bucket this CI job does not
provision.
"""
import ast
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HANDLER = REPO_ROOT / "src" / "handler.py"

FIXTURES = [
    "test_language_id_input.json",
    "test_language_classify_input.json",
    "test_translate_input.json",
    "test_summary_input.json",
]


def run_fixture(fixture_name: str) -> str | None:
    """Run one fixture; return an error message, or None on success."""
    test_input = (REPO_ROOT / fixture_name).read_text()

    proc = subprocess.run(
        [sys.executable, str(HANDLER), f"--test_input={test_input}"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    print(f"--- {fixture_name} ---")
    print(proc.stdout)
    print(proc.stderr, file=sys.stderr)

    if proc.returncode != 0:
        return f"{fixture_name}: handler process exited with code {proc.returncode}"

    result_line = next(
        (line for line in proc.stdout.splitlines() if "Job result:" in line),
        None,
    )
    if result_line is None:
        return f"{fixture_name}: no 'Job result' line in handler output"

    job_result = ast.literal_eval(result_line.split("Job result:", 1)[1].strip())
    output = job_result.get("output", job_result)

    if isinstance(output, dict) and "Error" in output:
        return f"{fixture_name}: task returned an error: {output['Error']}"

    print(f"OK  {fixture_name} -> {output}")
    return None


if __name__ == "__main__":
    failures = [msg for name in FIXTURES if (msg := run_fixture(name))]

    if failures:
        print("\nFAILED:")
        for msg in failures:
            print(f"  - {msg}")
        sys.exit(1)

    print("\nall fixtures passed")
