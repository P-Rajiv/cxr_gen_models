import csv
from pathlib import Path
from PIL import Image
import torch
from transformers import AutoModelForCausalLM, AutoProcessor

MODEL_ID = "microsoft/maira-2"
HF_HOME = "/data4/scratch/rajiv/huggingface_hub"
ROOT_DIR = Path("/data3/scratch/rajiv/cxr_gen_thesis/cxr_gen_diff/outputs/roentgen")
OUTPUT_CSV = ROOT_DIR / "maira_reports_simple.csv"

BATCH_SIZE = 16   # try 2 first, then 4, then 8

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


import torch
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

def generate_reports(image_paths):
    sample_inputs = []
    prompt_lengths = []

    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")

        single_input = processor.format_and_preprocess_reporting_input(
            current_frontal=image,
            current_lateral=None,
            prior_frontal=None,
            indication="Dyspnea.",
            technique="PA view of the chest.",
            comparison="None.",
            prior_report=None,
            return_tensors="pt",
            get_grounding=False,
        )

        sample_inputs.append(single_input)
        prompt_lengths.append(single_input["input_ids"].shape[1])

    # pad text inputs
    input_ids_list = [x["input_ids"].squeeze(0) for x in sample_inputs]
    attention_mask_list = [x["attention_mask"].squeeze(0) for x in sample_inputs]

    batch_input_ids = pad_sequence(
        input_ids_list,
        batch_first=True,
        padding_value=processor.tokenizer.pad_token_id
    )

    batch_attention_mask = pad_sequence(
        attention_mask_list,
        batch_first=True,
        padding_value=0
    )

    batch = {
        "input_ids": batch_input_ids.to(device),
        "attention_mask": batch_attention_mask.to(device),
    }

    # stack all remaining tensor keys if present and shapes match
    for key in sample_inputs[0].keys():
        if key in ["input_ids", "attention_mask"]:
            continue

        value0 = sample_inputs[0][key]

        if torch.is_tensor(value0):
            try:
                vals = [x[key].squeeze(0) for x in sample_inputs]
                batch[key] = torch.stack(vals, dim=0).to(device)
            except Exception as e:
                raise RuntimeError(f"Could not batch key '{key}': {e}")

    with torch.inference_mode():
        outputs = model.generate(
            **batch,
            max_new_tokens=300,
            use_cache=True,
            do_sample=False,
            num_beams=1,
        )

    reports = []
    for i in range(outputs.shape[0]):
        decoded = processor.decode(
            outputs[i][prompt_lengths[i]:],
            skip_special_tokens=True
        ).lstrip()

        parsed = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)

        if isinstance(parsed, str):
            reports.append(parsed.strip())
        else:
            reports.append(str(parsed).strip())

    return reports


all_image_paths = sorted(ROOT_DIR.rglob("*.jpg"))[:160]
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
        print(image_path.name,'-', report[:50])

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