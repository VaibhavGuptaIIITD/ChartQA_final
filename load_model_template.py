# ================================================================
# Generic HuggingFace Model Loader — GPU Server Template
#
# IMPORTANT: Always pass cache_dir on the GPU server.
# Without it, weights download to your home directory
# which has a tiny quota and will fill up immediately.
# ================================================================

from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "bert-base-cased"  # Replace with any HuggingFace model ID
CACHE_DIR = "/NS/ssdecl/work"   # Shared network storage on GPU server
                                 # Set to None when running on Colab/Kaggle

# Load tokenizer — weights cached to CACHE_DIR
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)

# Load model — same cache directory
model = AutoModel.from_pretrained(
    MODEL_NAME,
    cache_dir=CACHE_DIR
)

# Check GPU availability and move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # Should print "cuda" on the server
model = model.to(device)
