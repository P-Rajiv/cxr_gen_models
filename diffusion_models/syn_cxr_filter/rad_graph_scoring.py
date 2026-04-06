import pandas as pd
from radgraph import F1RadGraph

import os
os.environ["HF_HOME"]="/data4/scratch/rajiv/huggingface_hub"


# =========================
# CONFIG
# =========================
INPUT_CSV = "/data3/scratch/rajiv/cxr_gen_thesis/cxr_gen_diff/outputs/roentgen/maira_reports_simple.csv"
OUTPUT_CSV = "sample_radgraph_scores.csv"

IMAGE_COL = "image_path"
REF_COL = "prompt"         # reference
HYP_COL = "maira_report"   # hypothesis

MODEL_TYPE = "radgraph-xl"   # as shown in RadGraph README
BATCH_SIZE = 64              # adjust if memory is tight

# /data3/scratch/rajiv/cxr_gen_thesis/cxr_gen_diff/outputs/roentgen/atelectasis/100_1.jpg,
# The heart size is normal. The hilar and mediastinal contours are normal. The lungs are clear without evidence of focal consolidations concerning for pneumonia. There is no pleural effusion or pneumothorax. The visualized osseous structures are unremarkable


def get_reference_report(image_path):
    a, b = image_path.split("/")[-2:]
    # read b.split(_)[0] th line from 
    # /data3/scratch/rajiv/cxr_gen_thesis/syn_cxr_filter/diseasewise_prompts/{a}.txt
    with open(f"/data3/scratch/rajiv/cxr_gen_thesis/syn_cxr_filter/diseasewise_prompts/{a}.txt", "r") as f:
        lines = f.readlines()
        return lines[int(b.split("_")[0])].strip()


# =========================
# LOAD DATA
# =========================
df = pd.read_csv(INPUT_CSV)

df[REF_COL] = df[IMAGE_COL].apply(get_reference_report)

# keep original text safe
df[REF_COL] = df[REF_COL].fillna("").astype(str)
df[HYP_COL] = df[HYP_COL].fillna("").astype(str)

# optional: strip whitespace
df[REF_COL] = df[REF_COL].str.strip()
df[HYP_COL] = df[HYP_COL].str.strip()

# =========================
# INIT RADGRAPH SCORER
# =========================
f1radgraph = F1RadGraph(
    reward_level="all",
    model_type=MODEL_TYPE,
)

# =========================
# SCORE IN BATCHES
# =========================
rg_e_all = []
rg_er_all = []
rg_bar_er_all = []

n = len(df)

for start in range(0, n, BATCH_SIZE):
    end = min(start + BATCH_SIZE, n)

    refs = df.iloc[start:end][REF_COL].tolist()
    hyps = df.iloc[start:end][HYP_COL].tolist()

    # README API:
    # mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists
    mean_reward, reward_list, hyp_ann, ref_ann = f1radgraph(hyps=hyps, refs=refs)

    # reward_level="all" returns per-example lists for:
    # reward_list = (rg_e_list, rg_er_list, rg_bar_er_list)
    batch_rg_e, batch_rg_er, batch_rg_bar_er = reward_list

    rg_e_all.extend(batch_rg_e)
    rg_er_all.extend(batch_rg_er)
    rg_bar_er_all.extend(batch_rg_bar_er)

    print(f"Scored {end}/{n}")

# =========================
# SAVE OUTPUT
# =========================
df["radgraph_rg_e"] = rg_e_all
df["radgraph_rg_er"] = rg_er_all
df["radgraph_rg_bar_er"] = rg_bar_er_all

df.to_csv(OUTPUT_CSV, index=False)

print(f"\nSaved scored CSV to: {OUTPUT_CSV}")
print("\nDataset means:")
print("RG_E      :", df["radgraph_rg_e"].mean())
print("RG_ER     :", df["radgraph_rg_er"].mean())
print("RG_BAR_ER :", df["radgraph_rg_bar_er"].mean())