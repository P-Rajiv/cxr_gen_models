import csv
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image
from tqdm import tqdm

from medclip import MedCLIPModel, MedCLIPVisionModelViT, MedCLIPProcessor

# =========================================================
# Paths
# =========================================================
ROOT_DIR = Path("/data3/scratch/rajiv/cxr_gen_thesis/cxr_gen_diff/outputs/roentgen")
PROMPT_DIR = Path("/data3/scratch/rajiv/cxr_gen_thesis/syn_cxr_filter/diseasewise_prompts")
OUTPUT_CSV = ROOT_DIR / "medclip_scores_all.csv"

# =========================================================
# Settings
# =========================================================
BATCH_SIZE = 32
IMAGE_EXTS = {".jpg", ".jpeg", ".png"}

device = "cuda" if torch.cuda.is_available() else "cpu"

# =========================================================
# Prompt templates
# =========================================================
DISEASE_PROMPTS: Dict[str, List[str]] = {
    "atelectasis": [
        "atelectasis on chest x ray",
        "subsegmental atelectatic opacity on chest radiograph",
        "collapse of lung parenchyma on frontal chest x ray",
    ],
    "cardiomegaly": [
        "cardiomegaly on chest x ray",
        "enlarged cardiac silhouette on chest radiograph",
        "enlarged heart size on frontal chest x ray",
    ],
    "consolidation": [
        "consolidation on chest x ray",
        "focal airspace consolidation on chest radiograph",
        "alveolar opacity due to consolidation on frontal chest x ray",
    ],
    "edema": [
        "pulmonary edema on chest x ray",
        "diffuse pulmonary edema on chest radiograph",
        "interstitial or alveolar edema on frontal chest x ray",
    ],
    "enlarged_cardiomediastinum": [
        "enlarged cardiomediastinal silhouette on chest x ray",
        "widened cardiomediastinal contours on chest radiograph",
        "enlarged cardiomediastinum on frontal chest x ray",
    ],
    "fracture": [
        "fracture on chest x ray",
        "acute osseous fracture on chest radiograph",
        "rib fracture or bony fracture on frontal chest x ray",
    ],
    "lung_lesion": [
        "lung lesion on chest x ray",
        "focal pulmonary lesion on chest radiograph",
        "nodular or mass like lung lesion on frontal chest x ray",
    ],
    "lung_opacity": [
        "lung opacity on chest x ray",
        "pulmonary opacity on chest radiograph",
        "focal or diffuse lung opacity on frontal chest x ray",
    ],
    "pleural_effusion": [
        "pleural effusion on chest x ray",
        "pleural fluid collection on chest radiograph",
        "blunting of the costophrenic angle due to pleural effusion on frontal chest x ray",
    ],
    "pleural_other": [
        "pleural thickening or pleural plaque on chest x ray",
        "chronic pleural abnormality excluding pleural effusion and pneumothorax on chest radiograph",
        "pleural calcification pleural plaque or pleural thickening on frontal chest x ray",
    ],
    "pneumonia": [
        "pneumonia on chest x ray",
        "infectious airspace opacity on chest radiograph",
        "pulmonary infiltrate consistent with pneumonia on frontal chest x ray",
    ],
    "pneumothorax": [
        "pneumothorax on chest x ray",
        "pleural air and collapsed lung on chest radiograph",
        "pneumothorax on frontal chest x ray",
    ],
}

NORMAL_PROMPTS = [
    "normal chest x ray",
    "chest radiograph with no acute cardiopulmonary abnormality",
    "frontal chest x ray without focal airspace disease pleural effusion or pneumothorax",
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

def batched(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]

def move_inputs_to_device(inputs, device):
    out = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out

def encode_images(model, processor, images, device):
    inputs = processor(images=images, return_tensors="pt")
    inputs = move_inputs_to_device(inputs, device)
    with torch.no_grad():
        outputs = model(**inputs)
    img_embeds = outputs["img_embeds"]
    img_embeds = img_embeds / img_embeds.norm(dim=-1, keepdim=True)
    return img_embeds

def encode_texts(model, processor, texts, device):
    inputs = processor(text=texts, return_tensors="pt", padding=True)
    inputs = move_inputs_to_device(inputs, device)
    with torch.no_grad():
        outputs = model(**inputs)
    text_embeds = outputs["text_embeds"]
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    return text_embeds

# =========================================================
# Main
# =========================================================
def main():
    print(f"Using device: {device}")
    print("Loading MedCLIP...")

    processor = MedCLIPProcessor()
    model = MedCLIPModel(vision_cls=MedCLIPVisionModelViT)
    model.from_pretrained()
    model = model.to(device)
    model.eval()

    print("Encoding disease template prompts...")
    disease_template_features = {}
    for disease, prompts in DISEASE_PROMPTS.items():
        disease_template_features[disease] = encode_texts(model, processor, prompts, device)

    print("Encoding normal prompts...")
    normal_features = encode_texts(model, processor, NORMAL_PROMPTS, device)

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

            try:
                image_features = encode_images(model, processor, batch_images, device)
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

            try:
                prompt_features = encode_texts(model, processor, batch_used_prompts, device)
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

            rows = []
            for i, (img_path, used_prompt) in enumerate(zip(batch_valid_paths, batch_used_prompts)):
                disease = img_path.parent.name
                img_feat = image_features[i:i+1]
                prompt_feat = prompt_features[i:i+1]

                try:
                    prompt_score = float((img_feat @ prompt_feat.T).item())

                    if disease not in disease_template_features:
                        raise KeyError(f"No disease prompt templates defined for disease folder: {disease}")

                    d_feats = disease_template_features[disease]
                    disease_score = float((img_feat @ d_feats.T).max().item())

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