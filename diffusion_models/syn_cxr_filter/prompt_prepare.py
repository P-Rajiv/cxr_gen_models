#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------

DEFAULT_DISEASES = [
    "Atelectasis",
    "Cardiomegaly",
    "Consolidation",
    "Edema",
    "Enlarged Cardiomediastinum",
    "Fracture",
    "Lung Lesion",
    "Lung Opacity",
    "Pleural Effusion",
    "Pleural Other",
    "Pneumonia",
    "Pneumothorax",
]

CANONICAL_PROMPT_MAP = {
    "Atelectasis": "atelectasis",
    "Cardiomegaly": "cardiomegaly",
    "Consolidation": "consolidation",
    "Edema": "pulmonary edema",
    "Enlarged Cardiomediastinum": "enlarged cardiomediastinum",
    "Fracture": "fracture",
    "Lung Lesion": "lung lesion",
    "Lung Opacity": "lung opacity",
    "Pleural Effusion": "pleural effusion",
    "Pleural Other": "pleural abnormality",
    "Pneumonia": "pneumonia",
    "Pneumothorax": "pneumothorax",
}

SECTION_HEADER_RE = re.compile(r"(?m)^([A-Z][A-Z /-]{1,50}):\s*")


# -----------------------------
# Helpers
# -----------------------------

def normalize_id_series(s: pd.Series) -> pd.Series:
    return s.astype(str).str.replace(r"\.0$", "", regex=True).str.strip()


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_frontal_view(view: str) -> bool:
    if pd.isna(view):
        return False
    v = str(view).strip().upper()
    return v in {"PA", "AP"}


def view_priority(view: str) -> int:
    if pd.isna(view):
        return 99
    v = str(view).strip().upper()
    if v == "PA":
        return 0
    if v == "AP":
        return 1
    return 99


def build_report_path(mimic_root: Path, subject_id: str, study_id: str) -> Path:
    # Standard MIMIC-CXR report path:
    # files/p10/p10000032/s50414267.txt
    prefix = subject_id[:2]
    return mimic_root / "files" / f"p{prefix}" / f"p{subject_id}" / f"s{study_id}.txt"


def build_image_path(mimic_root: Path, subject_id: str, study_id: str, dicom_id: str) -> Path:
    # Standard image path:
    # files/p10/p10000032/s50414267/<dicom_id>.jpg
    prefix = subject_id[:2]
    return mimic_root / "files" / f"p{prefix}" / f"p{subject_id}" / f"s{study_id}" / f"{dicom_id}.jpg"


def read_report_text(report_path: Path) -> str:
    if not report_path.exists():
        return ""
    return report_path.read_text(encoding="utf-8", errors="ignore")


def parse_report_sections(report_text: str) -> Dict[str, str]:
    """
    Parse radiology report sections like:
    FINAL REPORT
    EXAMINATION:
    INDICATION:
    COMPARISON:
    FINDINGS:
    IMPRESSION:

    Returns a dict with lowercase keys, e.g.
    {
        "findings": "...",
        "impression": "...",
        "full_report": "..."
    }
    """
    text = report_text.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        return {}

    # Match uppercase section headers at line start, e.g. FINDINGS:
    section_re = re.compile(r"(?m)^\s*([A-Z][A-Z /-]{1,60})\s*:\s*")

    matches = list(section_re.finditer(text))
    if not matches:
        return {"full_report": clean_text(text)}

    sections = {}
    for i, match in enumerate(matches):
        header = match.group(1).strip().lower()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = clean_text(text[start:end])
        if content:
            sections[header] = content

    full_report = clean_text(text)
    if full_report:
        sections["full_report"] = full_report

    return sections


def choose_prompt_text(report_text: str, prompt_mode: str) -> str:
    """
    prompt_mode:
        - impression
        - findings
        - impression_or_findings
        - findings_or_impression
        - findings_plus_impression
        - full_report
    """
    sections = parse_report_sections(report_text)

    findings = sections.get("findings", "")
    impression = sections.get("impression", "")
    full_report = sections.get("full_report", "")

    if prompt_mode == "findings":
        return findings

    if prompt_mode == "impression":
        return impression

    if prompt_mode == "findings_or_impression":
        return findings if findings else impression

    if prompt_mode == "impression_or_findings":
        return impression if impression else findings

    if prompt_mode == "findings_plus_impression":
        if findings and impression:
            return f"FINDINGS: {findings} IMPRESSION: {impression}"
        return findings if findings else impression

    if prompt_mode == "full_report":
        return full_report

    return ""


def canonical_prompt(disease: str) -> str:
    return CANONICAL_PROMPT_MAP.get(disease, disease.lower())


def load_tables(mimic_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    meta_path = mimic_root / "mimic-cxr-2.0.0-metadata.csv"
    chexpert_path = mimic_root / "mimic-cxr-2.0.0-chexpert.csv"

    meta = pd.read_csv(meta_path)
    chex = pd.read_csv(chexpert_path)

    meta["subject_id"] = normalize_id_series(meta["subject_id"])
    meta["study_id"] = normalize_id_series(meta["study_id"])
    meta["dicom_id"] = normalize_id_series(meta["dicom_id"])

    chex["subject_id"] = normalize_id_series(chex["subject_id"])
    chex["study_id"] = normalize_id_series(chex["study_id"])

    return meta, chex


def choose_best_frontal_image(meta: pd.DataFrame, mimic_root: Path) -> pd.DataFrame:
    """
    Keep one frontal image per study. Prefer PA over AP.
    """
    frontal = meta[meta["ViewPosition"].apply(is_frontal_view)].copy()
    frontal["view_rank"] = frontal["ViewPosition"].apply(view_priority)

    frontal = frontal.sort_values(
        by=["subject_id", "study_id", "view_rank", "dicom_id"],
        ascending=[True, True, True, True]
    )

    best = frontal.groupby(["subject_id", "study_id"], as_index=False).first()

    best["image_path"] = best.apply(
        lambda row: str(build_image_path(
            mimic_root=mimic_root,
            subject_id=row["subject_id"],
            study_id=row["study_id"],
            dicom_id=row["dicom_id"],
        )),
        axis=1
    )

    best["report_path"] = best.apply(
        lambda row: str(build_report_path(
            mimic_root=mimic_root,
            subject_id=row["subject_id"],
            study_id=row["study_id"],
        )),
        axis=1
    )

    return best[
        ["subject_id", "study_id", "dicom_id", "ViewPosition", "image_path", "report_path"]
    ].copy()


def attach_prompt_text(df: pd.DataFrame, prompt_mode: str) -> pd.DataFrame:
    prompt_texts = []
    section_used = []

    for report_path_str in df["report_path"]:
        report_path = Path(report_path_str)
        report_text = read_report_text(report_path)
        prompt = choose_prompt_text(report_text, prompt_mode=prompt_mode)

        prompt_texts.append(prompt)

        sections = parse_report_sections(report_text)
        used = ""
        if prompt:
            for key in ["impression", "findings", "full_report"]:
                if sections.get(key, "") == prompt:
                    used = key
                    break
        section_used.append(used)

    out = df.copy()
    out["prompt_text"] = prompt_texts
    out["prompt_section"] = section_used
    out["prompt_len"] = out["prompt_text"].str.len().fillna(0).astype(int)
    return out


def prepare_master_table(
    mimic_root: Path,
    diseases: List[str],
    prompt_mode: str,
    exclude_support_devices: bool = True,
    require_report: bool = True,
) -> pd.DataFrame:
    meta, chex = load_tables(mimic_root)
    best_images = choose_best_frontal_image(meta, mimic_root)

    df = best_images.merge(
        chex,
        on=["subject_id", "study_id"],
        how="inner",
        validate="one_to_one"
    )

    for col in diseases + ["No Finding", "Support Devices"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = attach_prompt_text(df, prompt_mode=prompt_mode)

    if require_report:
        df = df[df["prompt_text"].astype(str).str.len() > 0].copy()

    if exclude_support_devices and "Support Devices" in df.columns:
        df = df[df["Support Devices"] != 1.0].copy()

    disease_cols_present = [d for d in diseases if d in df.columns]

    df["num_positive_diseases"] = (df[disease_cols_present] == 1.0).sum(axis=1)
    df["num_uncertain_diseases"] = (df[disease_cols_present] == -1.0).sum(axis=1)

    return df


def build_balanced_prompt_bank(
    master_df: pd.DataFrame,
    diseases: List[str],
    max_extra_positive: int = 0,
    drop_uncertain_anywhere: bool = True,
    min_prompt_len: int = 15,
    n_per_disease: Optional[int] = None,
    random_state: int = 42,
    dedupe_prompt_text: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build one balanced set where each selected study is assigned to one target disease.
    Default is very clean: single-disease positives only (max_extra_positive=0).
    """
    candidate_tables = []
    summary_rows = []

    for disease in diseases:
        if disease not in master_df.columns:
            print(f"[WARN] Disease column not found, skipping: {disease}")
            continue

        c = master_df.copy()

        # target disease positive
        c = c[c[disease] == 1.0].copy()

        # remove No Finding
        if "No Finding" in c.columns:
            c = c[c["No Finding"] != 1.0].copy()

        # remove uncertain target explicitly
        c = c[c[disease] != -1.0].copy()

        # optionally remove any uncertain disease among target disease set
        if drop_uncertain_anywhere:
            c = c[c["num_uncertain_diseases"] == 0].copy()

        # keep low-comorbidity cases
        c["extra_positive_diseases"] = c["num_positive_diseases"] - 1
        c = c[c["extra_positive_diseases"] <= max_extra_positive].copy()

        # require useful prompt text
        c = c[c["prompt_len"] >= min_prompt_len].copy()

        # assign the current target disease
        c["target_disease"] = disease
        c["canonical_target_prompt"] = canonical_prompt(disease)

        # optional prompt deduplication
        if dedupe_prompt_text:
            c = c.sort_values(
                by=["extra_positive_diseases", "prompt_len", "study_id"],
                ascending=[True, False, True]
            )
            c = c.drop_duplicates(subset=["prompt_text"]).copy()

        c = c.sort_values(
            by=["extra_positive_diseases", "prompt_len", "study_id"],
            ascending=[True, False, True]
        )

        summary_rows.append({
            "disease": disease,
            "available_clean_candidates": len(c),
        })

        candidate_tables.append(c)

    summary_df = pd.DataFrame(summary_rows)

    if summary_df.empty:
        raise RuntimeError("No valid disease candidates found.")

    if n_per_disease is None:
        n_per_disease = int(summary_df["available_clean_candidates"].min())

    if n_per_disease <= 0:
        raise RuntimeError("Balanced count is 0. Relax filters or reduce disease list.")

    print(f"[INFO] Using n_per_disease = {n_per_disease}")

    final_parts = []
    rng = np.random.RandomState(random_state)

    for c in candidate_tables:
        disease = c["target_disease"].iloc[0]

        if len(c) < n_per_disease:
            print(f"[WARN] {disease}: only {len(c)} candidates, skipping")
            continue

        sampled = c.sample(n=n_per_disease, random_state=rng.randint(0, 10_000_000)).copy()
        final_parts.append(sampled)

    if not final_parts:
        raise RuntimeError("No disease had enough samples after filtering.")

    final_df = pd.concat(final_parts, ignore_index=True)

    final_df = final_df[
        [
            "subject_id",
            "study_id",
            "dicom_id",
            "ViewPosition",
            "image_path",
            "report_path",
            "prompt_section",
            "prompt_text",
            "target_disease",
            "canonical_target_prompt",
            "num_positive_diseases",
            "extra_positive_diseases",
        ]
    ].sort_values(by=["target_disease", "study_id"]).reset_index(drop=True)

    selected_summary = (
        final_df.groupby("target_disease")
        .size()
        .reset_index(name="selected_count")
        .rename(columns={"target_disease": "disease"})
    )

    summary_df = summary_df.merge(selected_summary, on="disease", how="left").fillna(0)
    summary_df["selected_count"] = summary_df["selected_count"].astype(int)

    return final_df, summary_df


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Build a balanced MIMIC-CXR prompt bank.")
    parser.add_argument(
        "--mimic_root",
        type=str,
        required=True,
        help="Root folder containing MIMIC-CXR metadata CSVs and files/ directory."
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="balanced_prompt_bank.csv",
        help="Output CSV path."
    )
    parser.add_argument(
        "--summary_csv",
        type=str,
        default="balanced_prompt_bank_summary.csv",
        help="Output summary CSV path."
    )
    parser.add_argument(
        "--prompt_mode",
        type=str,
        default="impression_or_findings",
        choices=[
            "impression",
            "findings",
            "impression_or_findings",
            "findings_or_impression",
            "full_report",
        ],
        help="Which report section to use as prompt text."
    )
    parser.add_argument(
        "--diseases",
        nargs="+",
        default=DEFAULT_DISEASES,
        help="List of disease columns to balance."
    )
    parser.add_argument(
        "--n_per_disease",
        type=int,
        default=None,
        help="Samples per disease. Default: use minimum available clean count."
    )
    parser.add_argument(
        "--max_extra_positive",
        type=int,
        default=0,
        help="Max number of additional positive diseases allowed. 0 = single-disease only."
    )
    parser.add_argument(
        "--allow_uncertain_anywhere",
        action="store_true",
        help="If set, keep rows even if some non-target disease labels are uncertain (-1)."
    )
    parser.add_argument(
        "--min_prompt_len",
        type=int,
        default=15,
        help="Minimum prompt text length."
    )
    parser.add_argument(
        "--keep_support_devices",
        action="store_true",
        help="If set, do not remove studies with Support Devices = 1."
    )
    parser.add_argument(
        "--no_dedupe_prompt_text",
        action="store_true",
        help="If set, do not deduplicate identical prompt texts."
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed."
    )

    args = parser.parse_args()

    mimic_root = Path(args.mimic_root)

    master_df = prepare_master_table(
        mimic_root=mimic_root,
        diseases=args.diseases,
        prompt_mode=args.prompt_mode,
        exclude_support_devices=not args.keep_support_devices,
        require_report=True,
    )

    final_df, summary_df = build_balanced_prompt_bank(
        master_df=master_df,
        diseases=args.diseases,
        max_extra_positive=args.max_extra_positive,
        drop_uncertain_anywhere=not args.allow_uncertain_anywhere,
        min_prompt_len=args.min_prompt_len,
        n_per_disease=args.n_per_disease,
        random_state=args.random_state,
        dedupe_prompt_text=not args.no_dedupe_prompt_text,
    )

    final_df.to_csv(args.output_csv, index=False)
    summary_df.to_csv(args.summary_csv, index=False)

    print(f"[DONE] Saved prompt bank to: {args.output_csv}")
    print(f"[DONE] Saved summary to: {args.summary_csv}")
    print(summary_df.sort_values("disease").to_string(index=False))


if __name__ == "__main__":
    main()