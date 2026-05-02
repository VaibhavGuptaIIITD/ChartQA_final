# ================================================================
# ERROR ANALYSIS — Matcha and DePlot
# Goal  : Find patterns in model failures by chart type,
#         question type, and answer type.
# Input : matcha_errors.csv, deplot_errors.csv
# Output: error_by_chart_type.png, error_by_question_type.png
# ================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------------------
# STEP 1: Load error files produced by evaluation scripts
# -----------------------------------------------------------------
matcha_errors = pd.read_csv("matcha_errors.csv")
deplot_errors  = pd.read_csv("deplot_errors.csv")

print(f"Matcha errors : {len(matcha_errors)}")
print(f"DePlot errors : {len(deplot_errors)}")

# -----------------------------------------------------------------
# STEP 2: Infer chart type from image path
# ChartQA filenames are numeric (e.g. 5090.png), NOT descriptive.
# We fall back to checking the parent folder name, and if that also
# gives no signal we label it "unknown" so the analysis is honest.
# For a richer breakdown, join with the ChartQA metadata JSON
# (ChartQA Dataset/test/annotations/) which has an explicit
# "chart_type" field per sample.
# -----------------------------------------------------------------
def infer_chart_type(img_path):
    path = str(img_path).lower()
    # Check every component of the path (folder names may encode type)
    for part in path.replace("\\", "/").split("/"):
        if "bar"     in part: return "bar"
        if "line"    in part: return "line"
        if "pie"     in part: return "pie"
        if "dot"     in part: return "dot"
        if "scatter" in part: return "scatter"
    return "unknown"   # honest label — do not conflate with "other"

matcha_errors["chart_type"] = matcha_errors["image"].apply(infer_chart_type)
deplot_errors["chart_type"]  = deplot_errors["image"].apply(infer_chart_type)

# -----------------------------------------------------------------
# STEP 3: Infer question type from question text keywords
# -----------------------------------------------------------------
def infer_question_type(q):
    q = str(q).lower()
    if any(w in q for w in ["how many", "what is the number", "count"]):
        return "counting"
    if any(w in q for w in ["highest", "lowest", "maximum", "minimum", "most", "least"]):
        return "extremum"
    if any(w in q for w in ["difference", "more than", "less than", "compare", "between"]):
        return "comparison"
    if any(w in q for w in ["percentage", "percent", "%"]):
        return "percentage"
    if any(w in q for w in ["trend", "increase", "decrease", "change"]):
        return "trend"
    return "other"

matcha_errors["question_type"] = matcha_errors["question"].apply(infer_question_type)
deplot_errors["question_type"]  = deplot_errors["question"].apply(infer_question_type)

# -----------------------------------------------------------------
# STEP 4: Visualize failures by chart type
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

matcha_errors["chart_type"].value_counts().plot(
    kind="bar", ax=axes[0], color="steelblue", edgecolor="black"
)
axes[0].set_title("Matcha Errors by Chart Type")
axes[0].set_xlabel("Chart Type")
axes[0].set_ylabel("Number of Errors")

deplot_errors["chart_type"].value_counts().plot(
    kind="bar", ax=axes[1], color="coral", edgecolor="black"
)
axes[1].set_title("DePlot Errors by Chart Type")
axes[1].set_xlabel("Chart Type")

plt.tight_layout()
plt.savefig("error_by_chart_type.png", dpi=150)
plt.show()
print("[INFO] Saved error_by_chart_type.png")

# -----------------------------------------------------------------
# STEP 5: Visualize failures by question type
# -----------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

matcha_errors["question_type"].value_counts().plot(
    kind="bar", ax=axes[0], color="mediumseagreen", edgecolor="black"
)
axes[0].set_title("Matcha Errors by Question Type")

deplot_errors["question_type"].value_counts().plot(
    kind="bar", ax=axes[1], color="orchid", edgecolor="black"
)
axes[1].set_title("DePlot Errors by Question Type")

plt.tight_layout()
plt.savefig("error_by_question_type.png", dpi=150)
plt.show()
print("[INFO] Saved error_by_question_type.png")

# -----------------------------------------------------------------
# STEP 6: Summary stats — print top failure patterns
# -----------------------------------------------------------------
print("\n--- Matcha: Top chart-type + question-type combos ---")
print(
    matcha_errors.groupby(["chart_type", "question_type"])
    .size().sort_values(ascending=False).head(10)
)

print("\n--- DePlot: Top chart-type + question-type combos ---")
print(
    deplot_errors.groupby(["chart_type", "question_type"])
    .size().sort_values(ascending=False).head(10)
)
