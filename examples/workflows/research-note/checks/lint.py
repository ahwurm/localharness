"""Deterministic draft lint for the synthetic research-note example. Exit 0 = pass, 1 = fail."""
import re, sys
from pathlib import Path

BANNED = ["game-changer", "game changer", "revolutionary", "seamlessly", "cutting-edge", "leverage", "best-in-class",
          "turns every manager into an analyst", "connect any system in minutes", "80% less"]
MARKERS = ["TODO", "[citation needed]", "lorem ipsum"]

def main(path: str) -> int:
    text = Path(path).read_text(encoding="utf-8")
    body = "\n".join(line for line in text.splitlines() if not line.startswith("#"))
    problems = []
    low = body.lower()
    for phrase in BANNED:
        if phrase in low:
            problems.append(f"banned phrase: {phrase!r}")
    for marker in MARKERS:
        if marker.lower() in low:
            problems.append(f"placeholder marker: {marker!r}")
    for line in body.splitlines():
        if re.match(r"^\s*([-*•]|\d+\.)\s+", line):
            problems.append(f"bullet in body: {line.strip()[:40]!r}")
            break
    words = len(body.split())
    if not 350 <= words <= 600:
        problems.append(f"word count {words} outside 350-600")
    for sentence in re.split(r"(?<=[.!?])\s+", body.replace("\n", " ")):
        if len(sentence.split()) > 35:
            problems.append(f"sentence over 35 words: {sentence[:50]!r}")
            break
    if problems:
        print("FAIL " + path); [print(" - " + p) for p in problems]; return 1
    print(f"PASS {path} ({words} words)"); return 0

if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "draft.md"))
