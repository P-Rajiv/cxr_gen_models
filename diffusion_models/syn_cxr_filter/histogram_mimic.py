import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "No Finding",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
    "Support Devices",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--chexpert_csv",
        type=str,
        required=True,
        help="Path to mimic-cxr-2.0.0-chexpert.csv",
    )
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Directory to save outputs",
    )
    ap.add_argument(
        "--include_no_finding",
        action="store_true",
        help="Include No Finding in the histogram",
    )
    ap.add_argument(
        "--include_support_devices",
        action="store_true",
        help="Include Support Devices in the histogram",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.chexpert_csv)

    disease_cols = [c for c in DEFAULT_DISEASES if c in df.columns]

    if not args.include_no_finding and "No Finding" in disease_cols:
        disease_cols.remove("No Finding")

    if not args.include_support_devices and "Support Devices" in disease_cols:
        disease_cols.remove("Support Devices")

    summary = []
    for col in disease_cols:
        s = pd.to_numeric(df[col], errors="coerce")

        pos = int((s == 1.0).sum())
        unc = int((s == -1.0).sum())
        neg = int((s == 0.0).sum())
        missing = int(s.isna().sum())

        summary.append({
            "disease": col,
            "positive_count": pos,
            "uncertain_count": unc,
            "negative_count": neg,
            "missing_count": missing,
        })

    summary_df = pd.DataFrame(summary)
    summary_df = summary_df.sort_values("positive_count", ascending=False).reset_index(drop=True)

    summary_csv = out_dir / "mimic_disease_frequency_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    plt.figure(figsize=(12, 6))
    plt.bar(summary_df["disease"], summary_df["positive_count"])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("Positive count")
    plt.title("MIMIC-CXR disease frequencies (CheXpert labels)")
    plt.tight_layout()

    fig_path = out_dir / "mimic_disease_histogram.png"
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print(f"[DONE] Saved summary CSV: {summary_csv}")
    print(f"[DONE] Saved histogram PNG: {fig_path}")
    print("\nTop diseases by positive count:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()