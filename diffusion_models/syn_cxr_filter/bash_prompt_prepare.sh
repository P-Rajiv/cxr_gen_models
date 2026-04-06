# python prompt_prepare.py \
#   --mimic_root /data3/scratch/rajiv/mimic_cxr \
#   --output_csv balanced_prompt_bank_core8.csv \
#   --summary_csv balanced_prompt_bank_core8_summary.csv \
#   --prompt_mode impression_or_findings \
#   --max_extra_positive 0 \
#   --diseases \
#     Atelectasis Cardiomegaly Consolidation Edema \
#     "Lung Opacity" "Pleural Effusion" Pneumonia Pneumothorax

python prompt_prepare.py \
  --mimic_root /data3/scratch/rajiv/mimic_cxr \
  --output_csv balanced_prompt_bank.csv \
  --summary_csv balanced_prompt_bank_summary.csv \
  --prompt_mode findings \
  --max_extra_positive 0