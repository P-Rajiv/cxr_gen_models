import csv
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "microsoft/maira-2"
HF_HOME = "/data4/scratch/rajiv/huggingface_hub"
ROOT_DIR = Path("/data3/scratch/rajiv/cxr_gen_thesis/cxr_gen_diff/outputs/roentgen")
OUTPUT_CSV = ROOT_DIR / "maira_reports_simple.csv"

BATCH_SIZE = 8   # try 2 first, then 4, then 8

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float16 if device.type == "cuda" else torch.float32

print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=HF_HOME,
    torch_dtype=dtype,
)
processor = AutoProcessor.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    cache_dir=HF_HOME,
    use_fast=False,
)
model = model.eval().to(device)


def chunk_list(lst, batch_size):
    for i in range(0, len(lst), batch_size):
        yield lst[i:i + batch_size]


def generate_reports(image_paths):
    images = [Image.open(image_path).convert("RGB") for image_path in image_paths]

    inputs = processor.format_and_preprocess_reporting_input(
        current_frontal=images,
        current_lateral=[None] * len(images),
        prior_frontal=[None] * len(images),
        indication=["Dyspnea."] * len(images),
        technique=["PA view of the chest."] * len(images),
        comparison=["None."] * len(images),
        prior_report=[None] * len(images),
        return_tensors="pt",
        get_grounding=False,
    )

    inputs = {k: v.to(device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            use_cache=True,
            do_sample=False,
            num_beams=1,
        )

    prompt_length = inputs["input_ids"].shape[1]
    reports = []

    for i in range(outputs.shape[0]):
        decoded = processor.decode(
            outputs[i][prompt_length:],
            skip_special_tokens=True
        ).lstrip()

        parsed = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)

        if isinstance(parsed, str):
            reports.append(parsed.strip())
        else:
            reports.append(str(parsed).strip())

    return reports


all_image_paths = sorted(ROOT_DIR.rglob("*.jpg"))[:10]
rows = []

for batch_paths in chunk_list(all_image_paths, BATCH_SIZE):
    # breakpoint()
    # try:
    batch_reports = generate_reports(batch_paths)

    for image_path, report in zip(batch_paths, batch_reports):
        rows.append({
            "image_path": str(image_path),
            "maira_report": report,
            "status": "ok",
            "error": "",
        })
            # print(image_path.name, "ok")

    # except Exception as e:
    #     err = str(e)
    #     for image_path in batch_paths:
    #         rows.append({
    #             "image_path": str(image_path),
    #             "maira_report": "",
    #             "status": "error",
    #             "error": err,
    #         })
    #         print(image_path.name, "error")

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["image_path", "maira_report", "status", "error"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Saved to {OUTPUT_CSV}")