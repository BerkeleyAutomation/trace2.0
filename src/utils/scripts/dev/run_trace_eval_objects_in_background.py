#!/usr/bin/env python3
import json
import os
import re
from datetime import datetime

import argparse

from decluttering.src.ablation_study_script import run_vision_pipeline
from decluttering.src.trace_eval_helper import percentages_from_trace_list


def sort_key(path: str):
    name = os.path.basename(path)
    if name == "img.png":
        return (0, 1)
    m = re.search(r"\d+", name)
    return (1, int(m.group(0)) if m else 10**9)


def main(img_dir, output_dir):
    
    out_root = os.path.join(
        str(output_dir),
        f"trace_eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(out_root, exist_ok=True)

    files = []
    for f in os.listdir(img_dir):
        if f == "img.png" or re.fullmatch(r"img\d+\.png", f):
            files.append(os.path.join(img_dir, f))
    files = sorted(files, key=sort_key)

    summary = []
    for p in files:
        name = os.path.splitext(os.path.basename(p))[0]
        output_dir = os.path.join(out_root, name)
        os.makedirs(output_dir, exist_ok=True)

        output = run_vision_pipeline(
            image_path=p,
            output_dir=output_dir,
            num_endpoints=4,
            viz=False,
        )

        eval_result = percentages_from_trace_list(
            output.get("trace_list", []),
            total_circles_per_wire=76,
            click=False,
            viz=False,
            img=None,
        )

        row = {
            "image": os.path.basename(p),
            "circle_indices": eval_result["circle_indices"],
            "percentages": eval_result["percentages"],
            "average_percentage": eval_result["average_percentage"],
        }
        summary.append(row)
        print(
            f"{row['image']}: avg={row['average_percentage']:.4f}, "
            f"per_trace={row['percentages']}"
        )

    summary_path = os.path.join(out_root, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved summary: {summary_path}")
    print(f"Per-image outputs under: {out_root}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--image")
    parser.add_argument("--output_dir")

    args = parser.parse_args()

    image = args.image
    output_dir = args.output_dir
    main(image, output_dir)

### Example 
# python /home/justinyu/multicable-decluttering/decluttering/scripts/dev/run_trace_eval_objects_in_background.py --image "/home/justinyu/multicable-decluttering/decluttering/DATA_IROS26/colored_cables_paper/tier_3/2026-02-25 16:17:40.100879_success_keep" --output_dir "/home/justinyu/multicable-decluttering/decluttering/DATA_IROS26/colored_cables_paper/tier_3/eval"
