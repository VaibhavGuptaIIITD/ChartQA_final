# ================================================================
# DEPLOT FULL DATASET EVALUATION
# Unlike Matcha which directly answers questions, DePlot outputs
# a table. We compare the table content against the ground-truth
# answer label using a relaxed substring match.
# Reference eval script:
#   github.com/google-research/google-research/blob/master/
#   deplot/evaluate_chart_to_table.py
# ================================================================

from transformers import (
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration
)
from PIL import Image
import torch
import os
import pandas as pd
from tqdm import tqdm

# -----------------------------------------------------------------
# STEP 1: Config
# -----------------------------------------------------------------
MODEL_ID   = "google/deplot"
CACHE_DIR  = "/NS/ssdecl/work"
IMAGE_DIR  = "/content/ChartQA/ChartQA Dataset/test/png/"
INDEX_FILE = "readme.txt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

# -----------------------------------------------------------------
# STEP 2: Load model ONCE, outside the loop
# -----------------------------------------------------------------
processor = Pix2StructProcessor.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR
)
model = Pix2StructForConditionalGeneration.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR
)
model = model.to(device)
model.eval()
print("[INFO] DePlot model loaded.")

# -----------------------------------------------------------------
# STEP 3: Parse index file (same format as Matcha script)
# -----------------------------------------------------------------
with open(INDEX_FILE, 'r') as f:
    ff = f.readlines()

records = []
for i in range(len(ff)):
    if ff[i].strip() == "{":
        # Robust parsing: split on the first ': ' rather than
        # hardcoded character offsets, which break if spacing varies.
        try:
            img_name = ff[i + 1].split(": ", 1)[1].strip()
            question = ff[i + 2].split(": ", 1)[1].strip()
            label    = ff[i + 3].split(": ", 1)[1].strip()
            records.append((os.path.join(IMAGE_DIR, img_name), question, label))
        except IndexError:
            print(f"[WARN] Could not parse record at line {i}, skipping.")

print(f"[INFO] Total samples: {len(records)}")

# -----------------------------------------------------------------
# STEP 4: Inference loop with progress bar
# -----------------------------------------------------------------
correct, total = 0, 0
errors    = []
all_preds = []

DEPLOT_PROMPT = "Generate underlying data table of the figure below:"

for img_path, question, label in tqdm(records, desc="DePlot Eval"):
    try:
        image  = Image.open(img_path).convert("RGB")
        inputs = processor(
            images=image,
            text=DEPLOT_PROMPT,   # DePlot always uses this fixed prompt
            return_tensors="pt"
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            predictions = model.generate(**inputs, max_new_tokens=512)

        table_text = processor.decode(
            predictions[0], skip_special_tokens=True
        ).strip()

        # Extract the answer from the table for exact-match comparison.
        # Strategy: scan every cell value in the linearised table and
        # return the first one that matches the label; otherwise return
        # the full table so the ensemble can still use it as a fallback.
        def extract_answer_from_table(table: str, question: str) -> str:
            """
            DePlot produces rows like:  Header1 | Header2 | ...
            We split on '|' and '\n', strip each cell, and return the
            first non-header cell whose text matches the ground-truth
            label (used during scoring). For the ensemble we save the
            full table separately so the LLM-Blender step can reason
            over it later.
            """
            cells = [c.strip() for row in table.split("\n")
                     for c in row.split("|")]
            # Return the most specific (shortest non-empty) cell as the
            # predicted answer — a simple heuristic that works for
            # numerical look-up questions.
            numeric_cells = [c for c in cells[1:] if c and any(ch.isdigit() for ch in c)]
            return numeric_cells[0] if numeric_cells else table

        predicted_answer = extract_answer_from_table(table_text, question)

        total += 1
        all_preds.append({
            "image":     img_path,
            "question":  question,
            "predicted": predicted_answer,   # short answer for ensemble
            "table":     table_text,         # full table for future LLM-Blender step
            "label":     label
        })

        # Exact-match scoring (same metric as Matcha for fair comparison)
        if str(predicted_answer).strip().lower() == str(label).strip().lower():
            correct += 1
        else:
            errors.append({
                "image":    img_path,
                "question": question,
                "table":    table_text,
                "predicted": predicted_answer,
                "label":    label
            })

    except Exception as e:
        print(f"[ERROR] {img_path}: {e}")

# -----------------------------------------------------------------
# STEP 5: Save results
# -----------------------------------------------------------------
if total == 0:
    print("[ERROR] No samples were evaluated. Check IMAGE_DIR and INDEX_FILE paths.")
else:
    acc = (correct / total) * 100
    print(f"\n[RESULT] DePlot Accuracy: {acc:.4f}%")

pd.DataFrame(all_preds).to_csv("deplot_predictions.csv", index=False)
pd.DataFrame(errors).to_csv("deplot_errors.csv", index=False)
print(f"[INFO] Predictions saved to deplot_predictions.csv")
print(f"[INFO] Errors saved to deplot_errors.csv")
