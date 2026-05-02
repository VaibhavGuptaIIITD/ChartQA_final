# ChartQA — DePlot, Matcha & Ensemble Methods

Visual Question Answering on chart data using Google's DePlot and Matcha models, with ensemble strategies and error analysis.

> **Authors:** Vaibhav Gupta, Ratnango Ghosh  
> **Supervisor:** Subhendu Khatuya  
> **Dataset:** [ChartQA Benchmark](https://arxiv.org/pdf/2203.10244)  
> **Models:** [Google DePlot](https://github.com/google-research/google-research/tree/master/deplot), [Google Matcha](https://arxiv.org/pdf/2403.12596v1)

---

---

## Project Structure

```
chartqa-repo/
├── README.md
├── requirements.txt
├── scripts/
│   └── setup_venv.sh          # GPU server virtual environment setup
└── src/
    ├── load_model_template.py  # Generic HuggingFace model loader
    ├── evaluate_matcha.py      # Full Matcha evaluation pipeline
    ├── evaluate_deplot.py      # DePlot full dataset evaluation
    ├── deplot_quick_test.py    # DePlot single-image quick test
    ├── error_analysis.py       # Error pattern analysis & visualisation
    └── ensemble.py             # DePlot + Matcha ensemble strategy
```

---

## Setup

### 1. Install Dependencies (Colab / Local)

```bash
pip install -r requirements.txt
```

### 2. GPU Server Setup

```bash
bash scripts/setup_venv.sh
```

> **GPU Server Rules:**
> - Always work inside your own virtual environment under `/NS/ssdecl/work/<YOUR_NAME>/`
> - Always pass `cache_dir='/NS/ssdecl/work'` when loading HuggingFace models
> - Ping the supervisor before any session longer than 30 minutes
> - Do not use the server for unrelated work — usage is monitored

---

## Usage

### Matcha — Full Evaluation

```bash
python src/evaluate_matcha.py
```

Runs the `google/matcha-chartqa` model over the full ChartQA test set (~6,000 samples).  
Outputs `matcha_errors.csv` with all failed predictions.

### DePlot — Quick Single-Image Test

```bash
python src/deplot_quick_test.py
```

Downloads one chart from the ChartQA validation set and prints the linearised data table.

### DePlot — Full Evaluation

```bash
python src/evaluate_deplot.py
```

Runs `google/deplot` over the full test set. Outputs `deplot_errors.csv`.

### Error Analysis

```bash
python src/error_analysis.py
```

Reads `matcha_errors.csv` and `deplot_errors.csv`, classifies failures by chart type and question type, and saves:
- `error_by_chart_type.png`
- `error_by_question_type.png`

### Ensemble (DePlot + Matcha)

```bash
python src/ensemble.py
```

Merges predictions from both models using majority voting with Matcha as tiebreaker.  
Outputs `ensemble_results.csv`.

---

## Architecture Overview

### Matcha Pipeline
```
Chart Image + Question  →  Pix2Struct (fine-tuned)  →  Answer
```

### DePlot Pipeline
```
Chart Image  →  DePlot (Pix2Struct)  →  Linearised Table  →  LLM  →  Answer
```

### Ensemble Pipeline
```
            ┌──→  Matcha  ──┐
Chart + Q   │               ├──→  Majority Vote  →  Final Answer
            └──→  DePlot  ──┘
```

---

## Future Work

- **LLM-Blender ensembling** — pairwise ranking instead of naive majority vote ([resource](https://blog.allenai.org/llm-blender-a-simple-ensemble-learning-framework-for-llms-9e4bc57af23e))
- **Improved chart-to-text** — explore mistral, llama2, llama3, olmo for richer chart descriptions
- **Better prompting** — structured instructions to improve DePlot's table output
- **Diffusion-based chart understanding** — T2I-Adapter / IP-Adapter exploration

---

## References

1. Masry et al. (2022). [ChartQA Benchmark](https://arxiv.org/pdf/2203.10244)
2. Liu et al. (2023). [DePlot](https://github.com/google-research/google-research/tree/master/deplot)
3. Liu et al. (2024). [MatCha](https://arxiv.org/pdf/2403.12596v1)
4. Sutskever et al. (2014). [Seq2Seq Learning](https://proceedings.neurips.cc/paper/2014/file/a14ac55a4f27472c5d894ec1c3c743d2-Paper.pdf)
5. Jiang et al. (2023). [LLM-Blender](https://blog.allenai.org/llm-blender-a-simple-ensemble-learning-framework-for-llms-9e4bc57af23e)
