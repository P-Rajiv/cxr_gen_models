import pandas as pd
from radgraph import F1RadGraph

# generation_manifest.csv should contain at least:
# image_path, prompt_text, maira2_report
df = pd.read_csv("generation_manifest_with_maira2.csv")

refs = df["prompt_text"].fillna("").astype(str).tolist()
hyps = df["maira2_report"].fillna("").astype(str).tolist()

f1radgraph = F1RadGraph(
    reward_level="all",
    model_type="radgraph-xl"
)

mean_reward, reward_list, hyp_ann_lists, ref_ann_lists = f1radgraph(
    hyps=hyps,
    refs=refs
)

rg_e, rg_er, rg_bar_er = mean_reward
print("Mean rewards:")
print("RG_E   :", rg_e)
print("RG_ER  :", rg_er)
print("RG_BAR :", rg_bar_er)

df["radgraph_reward_all"] = reward_list
df.to_csv("generation_manifest_with_radgraph_scores.csv", index=False)


# from radgraph import RadGraph
# radgraph = RadGraph(model_type="modern-radgraph-xl")
# annotations = radgraph(["No evidence of pneumothorax following chest tube removal."])


# from radgraph import F1RadGraph
# refs = ["no acute cardiopulmonary abnormality",
#         "endotracheal tube is present and bibasilar opacities likely represent mild atelectasis",
# ]

# hyps = ["no acute cardiopulmonary abnormality",
#         "et tube terminates 2 cm above the carina and bibasilar opacities"
# ]
# f1radgraph = F1RadGraph(reward_level="all", model_type="radgraph-xl")
# mean_reward, reward_list, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=hyps, refs=refs)

# rg_e, rg_er, rg_bar_er = mean_reward

# print(mean_reward)

# import json
# from radgraph import get_radgraph_processed_annotations, RadGraph

# report = """
# Mild pulmonary edema with probable small bilateral pleural effusions.  
# More focal opacities at lung bases may reflect atelectasis but infection cannot be completely excluded.
# """
# model_type = "modern-radgraph-xl"
# radgraph = RadGraph(model_type=model_type)
# annotations = radgraph(
#     [report]
#     )

# processed_annotations = get_radgraph_processed_annotations(annotations)
# for annotation in processed_annotations["processed_annotations"]:
#     located_at = f" [Location: {', '.join(annotation['located_at'])}]" if annotation["located_at"] else ""
#     suggestive_of = f" [Suggestive of: {', '.join(annotation['suggestive_of'])}]" if annotation["suggestive_of"] else ""
#     tag = f" [Tag: {annotation['tags'][0]}]"
#     print(f"Observation: {annotation['observation']}{located_at}{suggestive_of}{tag}")