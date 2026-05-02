# ================================================================
# DEPLOT SINGLE IMAGE QUICK TEST
# Loads one chart from the ChartQA validation set via URL and
# converts it to a linearized data table.
# ================================================================

from transformers import (
    Pix2StructProcessor,
    Pix2StructForConditionalGeneration
)
import requests
from PIL import Image
import torch

# -----------------------------------------------------------------
# STEP 1: Load DePlot model (do this only once in production)
# -----------------------------------------------------------------
MODEL_ID  = "google/deplot"
CACHE_DIR = "/NS/ssdecl/work"  # Use None on Colab/Kaggle

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {device}")

processor = Pix2StructProcessor.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR
)
model = Pix2StructForConditionalGeneration.from_pretrained(
    MODEL_ID, cache_dir=CACHE_DIR
)
model = model.to(device)
model.eval()

# -----------------------------------------------------------------
# STEP 2: Load a chart image from ChartQA validation set via URL
# (Replace with Image.open(local_path) for local files)
# -----------------------------------------------------------------
url = (
    "https://raw.githubusercontent.com/vis-nlp/ChartQA/"
    "main/ChartQA%20Dataset/val/png/5090.png"
)
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# -----------------------------------------------------------------
# STEP 3: Run DePlot inference
# The fixed prompt always asks DePlot to output a data table,
# not to answer the question directly.
# -----------------------------------------------------------------
DEPLOT_PROMPT = "Generate underlying data table of the figure below:"

inputs = processor(
    images=image,
    text=DEPLOT_PROMPT,
    return_tensors="pt"
)
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    predictions = model.generate(**inputs, max_new_tokens=512)

table_output = processor.decode(predictions[0], skip_special_tokens=True)
print("[DePlot Output — Linearized Table]:")
print(table_output)

# Example output:
# Year | Revenue | Profit
# 2018 | 45.3    | 12.1
# 2019 | 52.7    | 15.4
# ...
