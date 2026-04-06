import os
import csv
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from PIL import Image
from tqdm import tqdm
import open_clip

# =========================================================
# Paths
# =========================================================
ROOT_DIR = Path("/data3/scratch/rajiv/cxr_gen_thesis/cxr_gen_diff/outputs/roentgen")
PROMPT_DIR = Path("/data3/scratch/rajiv/cxr_gen_thesis/syn_cxr_filter/diseasewise_prompts")
OUTPUT_CSV = ROOT_DIR / "biomedclip_scores_all.csv"
os.environ["HF_HOME"]="/data4/scratch/rajiv/huggingface_hub"

# =========================================================
# Settings
# =========================================================
HF_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

BATCH_SIZE = 128   # start with 32 if unsure, then try 64 or 128
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Prompt templates
# =========================================================
DISEASE_PROMPTS: Dict[str, List[str]] = {
    "atelectasis": [
        "a chest x-ray showing atelectasis",
        "a chest radiograph with subsegmental atelectatic opacity",
        "a frontal chest x-ray demonstrating collapse of lung parenchyma",
    ],
    "cardiomegaly": [
        "a chest x-ray showing cardiomegaly",
        "a chest radiograph with enlarged cardiac silhouette",
        "a frontal chest x-ray demonstrating enlarged heart size",
    ],
    "consolidation": [
        "a chest x-ray showing consolidation",
        "a chest radiograph with focal airspace consolidation",
        "a frontal chest x-ray demonstrating alveolar opacity due to consolidation",
    ],
    "edema": [
        "a chest x-ray showing pulmonary edema",
        "a chest radiograph with diffuse pulmonary edema",
        "a frontal chest x-ray demonstrating interstitial or alveolar edema",
    ],
    "enlarged_cardiomediastinum": [
        "a chest x-ray showing enlarged cardiomediastinal silhouette",
        "a chest radiograph with widened cardiomediastinal contours",
        "a frontal chest x-ray demonstrating enlarged cardiomediastinum",
    ],
    "fracture": [
        "a chest x-ray showing fracture",
        "a chest radiograph with acute osseous fracture",
        "a frontal chest x-ray demonstrating rib fracture or bony fracture",
    ],
    "lung_lesion": [
        "a chest x-ray showing lung lesion",
        "a chest radiograph with focal pulmonary lesion",
        "a frontal chest x-ray demonstrating nodular or mass-like lung lesion",
    ],
    "lung_opacity": [
        "a chest x-ray showing lung opacity",
        "a chest radiograph with pulmonary opacity",
        "a frontal chest x-ray demonstrating focal or diffuse lung opacity",
    ],
    "pleural_effusion": [
        "a chest x-ray showing pleural effusion",
        "a chest radiograph with pleural fluid collection",
        "a frontal chest x-ray demonstrating blunting of the costophrenic angle due to pleural effusion",
    ],
    "pleural_other": [
        "a chest x-ray showing pleural thickening or pleural plaque",
        "a chest radiograph with chronic pleural abnormality excluding pleural effusion and pneumothorax",
        "a frontal chest x-ray demonstrating pleural calcification, pleural plaque, or pleural thickening",
    ],
    "pneumonia": [
        "a chest x-ray showing pneumonia",
        "a chest radiograph with infectious airspace opacity",
        "a frontal chest x-ray demonstrating pulmonary infiltrate consistent with pneumonia",
    ],
    "pneumothorax": [
        "a chest x-ray showing pneumothorax",
        "a chest radiograph with pleural air and collapsed lung",
        "a frontal chest x-ray demonstrating pneumothorax",
    ],
}

NORMAL_PROMPTS = [
    "a normal chest x-ray",
    "a chest radiograph with no acute cardiopulmonary abnormality",
    "a frontal chest x-ray without focal airspace disease, pleural effusion, or pneumothorax",
]

# =========================================================
# Helpers
# =========================================================
def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS

def get_all_images(root_dir: Path) -> List[Path]:
    image_paths = []
    for disease_dir in sorted(root_dir.iterdir()):
        if not disease_dir.is_dir():
            continue
        for p in sorted(disease_dir.iterdir()):
            if p.is_file() and is_image_file(p):
                image_paths.append(p)
    return image_paths

def get_reference_prompt(image_path: Path) -> str:
    disease = image_path.parent.name
    fname = image_path.name

    # example: 100_10.jpg -> idx = 100
    idx = int(fname.split("_")[0])

    prompt_file = PROMPT_DIR / f"{disease}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")

    with open(prompt_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    if idx < 0 or idx >= len(lines):
        raise IndexError(
            f"Prompt index {idx} out of range for {prompt_file} with {len(lines)} lines"
        )

    return lines[idx].strip()

def safe_open_image(path: Path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open image {path}: {e}")
        return None

def batched(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def encode_texts(model, tokenizer, texts: List[str], device: str) -> torch.Tensor:
    tokens = tokenizer(texts).to(device)
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats

# =========================================================
# Main
# =========================================================
def main():
    print(f"Using device: {device}")
    print("Loading BiomedCLIP...")

    HF_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    model, preprocess = open_clip.create_model_from_pretrained(
        HF_MODEL_ID,
        device=device,
    )
    tokenizer = open_clip.get_tokenizer(HF_MODEL_ID)
    model.eval()

    print("Encoding disease template prompts...")
    disease_template_features = {}
    for disease, prompts in DISEASE_PROMPTS.items():
        disease_template_features[disease] = encode_texts(model, tokenizer, prompts, device)

    print("Encoding normal prompts...")
    normal_features = encode_texts(model, tokenizer, NORMAL_PROMPTS, device)

    image_paths = get_all_images(ROOT_DIR)
    print(f"Found {len(image_paths)} images")

    fieldnames = [
        "image_path",
        "disease",
        "used_prompt",
        "prompt_score",
        "disease_score",
        "normal_score",
        "disease_minus_normal",
        "status",
        "error",
    ]

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for batch_paths in tqdm(list(batched(image_paths, BATCH_SIZE)), desc="Scoring"):
            batch_images = []
            batch_valid_paths = []
            batch_used_prompts = []

            # -------------------------------------------------
            # Load images + actual used prompt
            # -------------------------------------------------
            for img_path in batch_paths:
                try:
                    img = safe_open_image(img_path)
                    if img is None:
                        writer.writerow({
                            "image_path": str(img_path),
                            "disease": img_path.parent.name,
                            "used_prompt": "",
                            "prompt_score": "",
                            "disease_score": "",
                            "normal_score": "",
                            "disease_minus_normal": "",
                            "status": "error",
                            "error": "image_open_failed",
                        })
                        continue

                    used_prompt = get_reference_prompt(img_path)

                    batch_images.append(img)
                    batch_valid_paths.append(img_path)
                    batch_used_prompts.append(used_prompt)

                except Exception as e:
                    writer.writerow({
                        "image_path": str(img_path),
                        "disease": img_path.parent.name,
                        "used_prompt": "",
                        "prompt_score": "",
                        "disease_score": "",
                        "normal_score": "",
                        "disease_minus_normal": "",
                        "status": "error",
                        "error": str(e),
                    })

            if len(batch_valid_paths) == 0:
                continue

            # -------------------------------------------------
            # Encode image batch
            # -------------------------------------------------
            try:
                image_tensors = torch.stack([preprocess(img) for img in batch_images]).to(device)

                with torch.no_grad(), torch.cuda.amp.autocast(enabled=(device == "cuda")):
                    image_features = model.encode_image(image_tensors)
                    image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            except Exception as e:
                for img_path, used_prompt in zip(batch_valid_paths, batch_used_prompts):
                    writer.writerow({
                        "image_path": str(img_path),
                        "disease": img_path.parent.name,
                        "used_prompt": used_prompt,
                        "prompt_score": "",
                        "disease_score": "",
                        "normal_score": "",
                        "disease_minus_normal": "",
                        "status": "error",
                        "error": f"image_encoding_failed: {e}",
                    })
                continue

            # -------------------------------------------------
            # Encode actual prompts for this batch
            # -------------------------------------------------
            try:
                prompt_features = encode_texts(model, tokenizer, batch_used_prompts, device)
            except Exception as e:
                for img_path, used_prompt in zip(batch_valid_paths, batch_used_prompts):
                    writer.writerow({
                        "image_path": str(img_path),
                        "disease": img_path.parent.name,
                        "used_prompt": used_prompt,
                        "prompt_score": "",
                        "disease_score": "",
                        "normal_score": "",
                        "disease_minus_normal": "",
                        "status": "error",
                        "error": f"prompt_encoding_failed: {e}",
                    })
                continue

            # -------------------------------------------------
            # Compute scores per sample
            # -------------------------------------------------
            rows = []
            for i, (img_path, used_prompt) in enumerate(zip(batch_valid_paths, batch_used_prompts)):
                disease = img_path.parent.name
                img_feat = image_features[i:i+1]           # [1, D]
                prompt_feat = prompt_features[i:i+1]       # [1, D]

                try:
                    # actual used prompt similarity
                    prompt_score = float((img_feat @ prompt_feat.T).item())

                    # disease template similarity
                    if disease not in disease_template_features:
                        raise KeyError(f"No disease prompt templates defined for disease folder: {disease}")

                    d_feats = disease_template_features[disease]  # [K, D]
                    disease_score = float((img_feat @ d_feats.T).max().item())

                    # normal similarity
                    normal_score = float((img_feat @ normal_features.T).max().item())

                    disease_minus_normal = disease_score - normal_score

                    rows.append({
                        "image_path": str(img_path),
                        "disease": disease,
                        "used_prompt": used_prompt,
                        "prompt_score": prompt_score,
                        "disease_score": disease_score,
                        "normal_score": normal_score,
                        "disease_minus_normal": disease_minus_normal,
                        "status": "ok",
                        "error": "",
                    })

                except Exception as e:
                    rows.append({
                        "image_path": str(img_path),
                        "disease": disease,
                        "used_prompt": used_prompt,
                        "prompt_score": "",
                        "disease_score": "",
                        "normal_score": "",
                        "disease_minus_normal": "",
                        "status": "error",
                        "error": str(e),
                    })

            writer.writerows(rows)

    print(f"Saved CSV to: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()