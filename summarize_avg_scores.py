#!/usr/bin/env python3
"""
Collect 'score avg' values from all models under objdet_results/t2i and print:
<model_folder_name> & <score_rounded_to_7_decimals>
"""

from pathlib import Path
import argparse
import re
import sys
from typing import Optional

# Matches lines like: "score avg:0.31906860526395125"
SCORE_RE = re.compile(r"score\s*avg\s*:\s*([+-]?\d+(?:\.\d+)?)", re.IGNORECASE)

def read_score(avg_file: Path) -> Optional[float]:
    try:
        text = avg_file.read_text(encoding="utf-8", errors="ignore")
    except FileNotFoundError:
        return None
    m = SCORE_RE.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None

def main(t2i_dir: Path, out_path: Optional[Path]) -> None:
    if not t2i_dir.is_dir():
        sys.exit(f"Base dir not found: {t2i_dir.resolve()}")

    lines = []
    for model_dir in sorted(p for p in t2i_dir.iterdir() if p.is_dir()):
        avg_file = model_dir / "labels" / "annotation_obj_detection_2d" / "avg_score.txt"
        score = read_score(avg_file)
        if score is None:
            # Write issues to stderr so stdout stays clean if you're piping to a file.
            print(f"# Skipped: {model_dir.name} (missing or unparsable {avg_file})", file=sys.stderr)
            continue
        lines.append(f"{model_dir.name} , {score:.5f}")

    output = "\n".join(lines) + ("\n" if lines else "")
    if out_path:
        out_path.write_text(output, encoding="utf-8")
    else:
        print(output, end="")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Summarize avg_score.txt values for all models under objdet_results/t2i."
    )
    parser.add_argument(
        "t2i_dir",
        nargs="?",
        default="objdet_results/t2i",
        help="Path to the t2i folder (default: objdet_results/t2i)",
    )
    parser.add_argument(
        "-o", "--out",
        type=Path,
        help="Optional output file (if omitted, prints to stdout)",
    )
    args = parser.parse_args()
    main(Path(args.t2i_dir), args.out)
