import argparse
from pathlib import Path

import pandas as pd


def sanitize_name(text: str) -> str:
    text = str(text).strip().lower()
    text = text.replace("/", "_")
    text = text.replace("-", "_")
    text = text.replace(" ", "_")
    text = "_".join(filter(None, text.split("_")))
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_csv", type=str, required=True,
                    help="Balanced prompt bank CSV")
    ap.add_argument("--out_dir", type=str, required=True,
                    help="Output folder for disease-wise txt files")
    ap.add_argument("--prompt_col", type=str, default="prompt_text",
                    help="Column containing prompts")
    ap.add_argument("--disease_col", type=str, default="target_disease",
                    help="Column containing disease labels")
    ap.add_argument("--dedupe", action="store_true",
                    help="Remove duplicate prompts within each disease")
    args = ap.parse_args()

    input_csv = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_csv)

    if args.prompt_col not in df.columns:
        raise ValueError(f"Missing prompt column: {args.prompt_col}")
    if args.disease_col not in df.columns:
        raise ValueError(f"Missing disease column: {args.disease_col}")

    df = df.dropna(subset=[args.prompt_col, args.disease_col]).copy()
    df[args.prompt_col] = df[args.prompt_col].astype(str).str.strip()
    df[args.disease_col] = df[args.disease_col].astype(str).str.strip()
    df = df[df[args.prompt_col] != ""].copy()

    summary_rows = []

    for disease, subdf in df.groupby(args.disease_col):
        prompts = subdf[args.prompt_col].tolist()

        if args.dedupe:
            seen = set()
            deduped = []
            for p in prompts:
                if p not in seen:
                    seen.add(p)
                    deduped.append(p)
            prompts = deduped

        fname = sanitize_name(disease) + ".txt"
        out_path = out_dir / fname

        with open(out_path, "w", encoding="utf-8") as f:
            for p in prompts:
                f.write(p.strip() + "\n")

        summary_rows.append({
            "disease": disease,
            "file_name": fname,
            "num_prompts": len(prompts),
            "path": str(out_path),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("disease")
    summary_csv = out_dir / "prompt_txt_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    print(f"[DONE] Wrote disease prompt files to: {out_dir}")
    print(f"[DONE] Summary CSV: {summary_csv}")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()