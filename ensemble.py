# ================================================================
# ENSEMBLE — DePlot + Matcha
# Strategy : Simple majority vote.
#   - Where both agree  → high confidence answer
#   - Where they differ → use Matcha as tiebreaker
#     (Matcha scored higher: 75.88% vs DePlot TBD)
#
# Extension: Replace tiebreaker with LLM-Blender pairwise ranker
# for a more principled selection.
# ================================================================

import pandas as pd

# -----------------------------------------------------------------
# STEP 1: Load pre-computed predictions from both models
# (Run evaluate_matcha.py and evaluate_deplot.py first)
# Expected columns: image, question, predicted, label
# -----------------------------------------------------------------
matcha_preds = pd.read_csv("matcha_predictions.csv")
deplot_preds  = pd.read_csv("deplot_predictions.csv")

# Merge on image + question so each row has both model outputs
merged = matcha_preds.merge(
    deplot_preds,
    on=["image", "question", "label"],
    suffixes=("_matcha", "_deplot")
)
print(f"[INFO] Merged dataset size: {len(merged)}")

# -----------------------------------------------------------------
# STEP 2: Apply ensemble decision rule
# -----------------------------------------------------------------
def ensemble_decision(row):
    """
    Decision rule:
      - If both agree  → return that answer (high confidence)
      - If they differ → return Matcha output as tiebreaker
        (Matcha had higher overall accuracy in solo evaluation)
    TODO: Replace with LLM-Blender pairwise ranking for a more
          principled selection between conflicting predictions.
    """
    pred_matcha = str(row["predicted_matcha"]).strip().lower()
    pred_deplot  = str(row["predicted_deplot"]).strip().lower()

    if pred_matcha == pred_deplot:
        return row["predicted_matcha"]   # Agreement: high confidence
    else:
        # Tiebreaker: use the stronger model (Matcha, 75.88%)
        return row["predicted_matcha"]

merged["ensemble_pred"] = merged.apply(ensemble_decision, axis=1)

# -----------------------------------------------------------------
# STEP 3: Score the ensemble
# -----------------------------------------------------------------
merged["correct_matcha"] = (
    merged["predicted_matcha"].str.strip().str.lower()
    == merged["label"].str.strip().str.lower()
)
merged["correct_deplot"] = (
    merged["predicted_deplot"].str.strip().str.lower()
    == merged["label"].str.strip().str.lower()
)
merged["correct_ensemble"] = (
    merged["ensemble_pred"].str.strip().str.lower()
    == merged["label"].str.strip().str.lower()
)

acc_matcha   = merged["correct_matcha"].mean() * 100
acc_deplot   = merged["correct_deplot"].mean() * 100
acc_ensemble = merged["correct_ensemble"].mean() * 100

print(f"\n[RESULT] Matcha Accuracy  : {acc_matcha:.2f}%")
print(f"[RESULT] DePlot Accuracy  : {acc_deplot:.2f}%")
print(f"[RESULT] Ensemble Accuracy: {acc_ensemble:.2f}%")

# -----------------------------------------------------------------
# STEP 4: Analysis — where does the ensemble beat solo models?
# -----------------------------------------------------------------
# Cases where ensemble is correct but Matcha was wrong
ensemble_wins = merged[
    merged["correct_ensemble"] & ~merged["correct_matcha"]
]
print(f"\n[INFO] Cases where ensemble beats Matcha : {len(ensemble_wins)}")

# Cases where ensemble is wrong but at least one model was right
ensemble_loses = merged[
    ~merged["correct_ensemble"]
    & (merged["correct_matcha"] | merged["correct_deplot"])
]
print(f"[INFO] Cases where ensemble loses to a solo model: {len(ensemble_loses)}")

# Save full results
merged.to_csv("ensemble_results.csv", index=False)
print("[INFO] Full results saved to ensemble_results.csv")
