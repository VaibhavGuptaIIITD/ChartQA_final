# ================================================================
# MATCHA CHARTQA EVALUATION SCRIPT
# Model  : google/matcha-chartqa (Pix2Struct fine-tuned)
# Task   : Direct Visual Question Answering on chart images
# Dataset: ChartQA test split (~6000 samples)
# Result : 75.88% exact-match accuracy (full run, ~17 hours)
# ================================================================

from transformers import (
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration
)
from PIL import Image
import torch
import os
import pandas as pd

# -----------------------------------------------------------------
# STEP 1: Configuration
# -----------------------------------------------------------------
MODEL_ID   = "google/matcha-chartqa"
CACHE_DIR  = "/NS/ssdecl/work"   # GPU server cache dir; use None on Colab/Kaggle
IMAGE_DIR  = "/content/ChartQA/ChartQA Dataset/test/png/"
INDEX_FILE = "readme.txt"         # ChartQA index file listing image, question, answer
MAX_TOKENS = 512                  # Max answer generation length
TEST_LIMIT = None                 # Set to an int (e.g. 60) for a quick smoke-test;
                                  # set to None to run the full test set

# -----------------------------------------------------------------
# STEP 2: Select compute device
# -----------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Running on: {device}")

# -----------------------------------------------------------------
# STEP 3: Load the model and processor ONCE (outside the loop!)
# Loading inside the loop re-downloads weights every iteration —
# a common mistake that makes evaluation 10x slower.
# -----------------------------------------------------------------
print(f"[INFO] Loading processor and model: {MODEL_ID}")
processor = Pix2StructProcessor.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR
)
model = Pix2StructForConditionalGeneration.from_pretrained(
    MODEL_ID,
    cache_dir=CACHE_DIR
)
model = model.to(device)  # Move weights to GPU
model.eval()              # Disable dropout / batch-norm updates
print("[INFO] Model loaded and moved to device.")

# -----------------------------------------------------------------
# STEP 4: Parse the ChartQA index file
# The readme.txt has a structured format where each record
# starts with a tab+brace line, followed by:
#   ff[i+1] = image filename
#   ff[i+2] = question
#   ff[i+3] = ground truth answer
# -----------------------------------------------------------------
print(f"[INFO] Reading index file: {INDEX_FILE}")
with open(INDEX_FILE, 'r') as f:
    ff = f.readlines()

records = []  # Will store (image_path, question, label) tuples

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

# Apply test limit if set (useful for quick debugging)
if TEST_LIMIT is not None:
    records = records[:TEST_LIMIT]
    print(f"[INFO] Limiting evaluation to first {TEST_LIMIT} samples.")

print(f"[INFO] Total samples to evaluate: {len(records)}")

# -----------------------------------------------------------------
# STEP 5: Run inference loop
# -----------------------------------------------------------------
correct = 0
total   = 0
errors  = []  # Store failed cases for later error analysis
all_preds = []

for idx, (img_path, question, label) in enumerate(records):
    try:
        # Load image from local path
        image = Image.open(img_path).convert("RGB")

        # Tokenize image patches + question text together
        # return_tensors="pt" gives PyTorch tensors
        inputs = processor(
            images=image,
            text=question,
            return_tensors="pt"
        )

        # Move input tensors to the same device as model
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Generate answer with no gradient tracking (saves memory)
        with torch.no_grad():
            predictions = model.generate(
                **inputs,
                max_new_tokens=MAX_TOKENS
            )

        # Decode token IDs back to a string; strip special tokens
        ans = processor.decode(
            predictions[0],
            skip_special_tokens=True
        ).strip()

        total += 1
        all_preds.append({"image": img_path, "question": question, "predicted": ans, "label": label})

        # Exact-match comparison
        if str(ans) == str(label).strip():
            correct += 1
        else:
            errors.append({
                "image":     img_path,
                "question":  question,
                "predicted": ans,
                "label":     label
            })

        # Print running accuracy every 50 samples
        if (idx + 1) % 50 == 0:
            running_acc = (correct / total) * 100
            print(f"[{idx+1}/{len(records)}] Running Accuracy: {running_acc:.2f}%")

    except FileNotFoundError:
        print(f"[WARN] Image not found: {img_path} — skipping.")
    except Exception as e:
        print(f"[ERROR] Sample {idx} failed: {e} — skipping.")

# -----------------------------------------------------------------
# STEP 6: Final accuracy and error export
# -----------------------------------------------------------------
if total == 0:
    print("[ERROR] No samples were evaluated. Check IMAGE_DIR and INDEX_FILE paths.")
else:
    final_accuracy = (correct / total) * 100
    print(f"\n[RESULT] Final Matcha Accuracy: {final_accuracy:.4f}%")
    print(f"[RESULT] Correct: {correct} / Total: {total}")

# Save all predictions for ensemble use
pd.DataFrame(all_preds).to_csv("matcha_predictions.csv", index=False)
print(f"[INFO] All predictions saved to matcha_predictions.csv")

# Save errors to CSV for offline analysis
df_errors = pd.DataFrame(errors)
df_errors.to_csv("matcha_errors.csv", index=False)
print(f"[INFO] {len(errors)} error cases saved to matcha_errors.csv")
