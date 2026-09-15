#!/usr/bin/env python3
"""Fallback C (ADR-0181 candidate): put the project-rule sentence in the write tools' content descriptions.

usage: adr0181_apply_toolrule.py <write_file.py> <edit.py>  (edits in place; prints the md5 before and after)
"""

import hashlib
import sys
from pathlib import Path

WRITE_OLD = '                    "description": "Full text content to write to the file.",\n'
WRITE_NEW = (
    '                    "description": (\n'
    '                        "Full text content to write to the file. Where the project context "\n'
    '                        "(the guides and conventions in your system prompt) states a rule "\n'
    '                        "that applies to this file, the content must follow it — even where "\n'
    '                        "files already in the project do otherwise."\n'
    "                    ),\n"
)
EDIT_OLD = '                    "description": "Text to insert in place of \'old_string\'.",\n'
EDIT_NEW = (
    '                    "description": (\n'
    "                        \"Text to insert in place of 'old_string'. Where the project context \"\n"
    '                        "(the guides and conventions in your system prompt) states a rule "\n'
    '                        "that applies to this file, the new text must follow it — even where "\n'
    '                        "files already in the project do otherwise."\n'
    "                    ),\n"
)


def md5(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def apply(path: Path, old: str, new: str, label: str) -> None:
    text = path.read_text(encoding="utf-8")
    before = md5(text)
    assert old in text, f"{label} anchor"
    text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")
    print(path.name, before, "->", md5(text))


if __name__ == "__main__":
    apply(Path(sys.argv[1]), WRITE_OLD, WRITE_NEW, "write_file")
    apply(Path(sys.argv[2]), EDIT_OLD, EDIT_NEW, "edit_file")
