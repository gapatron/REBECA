# REBECA

Personalized diffusion models that learn to generate **RE**commendations **BE**yond **CA**talogs.

<p align="center">
  <img src="notebooks/outputs/cvpr_heatmaps/posterior_prior_delta_heatmap.png" alt="REBECA heatmap preview" width="65%">
</p>

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Components](#key-components)
- [Repository Layout](#repository-layout)
- [Getting Started](#getting-started)
  - [Environment Setup](#environment-setup)
  - [Dataset Preparation](#dataset-preparation)
- [Training Pipelines](#training-pipelines)
  - [Diffusion Prior](#diffusion-prior)
  - [Neural Collaborative Filtering Baseline](#neural-collaborative-filtering-baseline)
  - [Grid Search & Model Selection](#grid-search--model-selection)
- [Evaluation & Diagnostics](#evaluation--diagnostics)
- [Ablations & Reproducibility](#ablations--reproducibility)
- [Notebooks](#notebooks)
- [Results & Figures](#results--figures)
- [Citation](#citation)
- [License](#license)

---

## Project Overview

REBECA explores recommender systems that **generate** personalized content rather than ranking items in a fixed catalog. The project couples:

1. A user-conditioned diffusion prior that models aesthetic preferences.
2. A neural collaborative filtering (NCF) backbone for calibration and scoring.
3. Comprehensive evaluation pipelines and CVPR-ready visualizations.

The default dataset is **FLICKR-AES**, augmented with human preference annotations. All components are modular so you can swap in other datasets or scoring heads.

---

## Key Components

- **Diffusion Prior (`prior_models.py`)**  
  Learns user-conditioned image embeddings with classifier-free guidance (CFG) dropout.

- **Personalized Sampler (`sampling.py`)**  
  Generates per-user image embeddings or full images with Stable Diffusion + IP-Adapter.

- **NCF Scorer (`ncf.py`)**  
  Trains a neural collaborative filtering model for posterior scoring, including grid-search wrappers.

- **Evaluation Suite (`evaluation.py`, notebooks)**  
  Computes ROC/AUC, bootstrap confidence intervals, and aesthetic metrics (HPSv2 & LAION).

- **Ablation Harness (`ablations.py`)**  
  Reproduces cross-CFG sweeps, prompt variations, and baseline generations.

---

## Repository Layout

```
REBECA/
├── data/                     # Raw, processed, and evaluation assets (FLICKR-AES)
├── notebooks/                # Analysis, diagnostics, and figure notebooks
│   ├── model_selection.ipynb # CFG sweeps, LAION scoring, CVPR heatmaps
│   ├── ncf_diagnostics.ipynb # NCF training diagnostics & bootstrap AUC
│   ├── evaluation.ipynb      # Aesthetic scorers (HPSv2, LAION) & persona evaluation
│   └── outputs/              # Exported figures (PNG/PDF) for the paper
├── prior_models.py           # Diffusion prior architectures
├── sampling.py               # Shared sampling utilities
├── train_priors.py           # Diffusion prior training entry point
├── grid_search.py            # Hyper-parameter sweeps for NCF
├── ncf.py                    # NCF model, training loop, and persistence helpers
├── evaluation.py             # Batch evaluation scripts
├── ablations.py              # Reproduction of cross-CFG and cross-prompt studies
├── utils.py                  # Shared helpers (seeding, serialization, etc.)
├── LICENSE                   # Apache 2.0
└── README.md                 # You are here
```

---

## Getting Started

### Environment Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-org>/REBECA.git
   cd REBECA
   ```

2. **Create a virtual environment** (conda recommended)
   ```bash
   conda create -n rebeca python=3.10
   conda activate rebeca
   ```

3. **Install core dependencies**
   ```bash
   pip install -r requirements.txt  # if available
   ```
   or install the main libraries manually:
   ```bash
   pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
   pip install diffusers==0.27.0 transformers==4.44.0 accelerate==0.33.0
   pip install pandas scikit-learn tqdm seaborn matplotlib ipywidgets
   ```

4. **Authenticate with Hugging Face (optional but recommended)**
   ```bash
   huggingface-cli login
   ```
   Required for certain Stable Diffusion weights.

### Dataset Preparation

1. Download the **FLICKR-AES** dataset and place the raw assets under `data/flickr/raw/`.
2. Preprocess embeddings and metadata using the available notebooks (`notebooks/lora_preprocessing.ipynb`, `notebooks/embeddings.ipynb`) or custom scripts.
3. Ensure the processed artifacts (CSV/NPZ/PT) match the paths expected by `train_priors.py`, `ncf.py`, and `evaluation.py`. See `data/flickr/processed/` for reference.

> **Tip:** Large binary artifacts (e.g., image embeddings) are stored with `torch.save(..., weights_only=True)` for compactness.

---

## Training Pipelines

### Diffusion Prior

Train the user-conditioned diffusion prior:

```bash
python train_priors.py \
  --config configs/prior_default.yaml \
  --output ./data/flickr/evaluation/diffusion_priors/models/weights/
```

Key features:
- Classifier-free guidance dropout on user IDs and score tokens.
- Support for noise/objective choices (`epsilon`, `sample`, `v_prediction`).
- Automatic early stopping with ReduceLROnPlateau.

Inspect training dynamics via `notebooks/train_priors.ipynb`.

### Neural Collaborative Filtering Baseline

Run the NCF training loop directly:

```python
from ncf import train_ncf
results = train_ncf(
    U=U_train, E=E_train, Y=Y_train,
    d=320,
    mlp_layers=[64, 32],
    lr=1e-4,
    device="cuda"
)
```

or use the `NCF` class with grid search:

```python
from ncf import NCF
ncf = NCF()
ncf.fit(U_train, E_train, Y_train, device="cuda")
```

Diagnostics:
- `notebooks/ncf_diagnostics.ipynb` (bootstrap AUC, ROC curves, calibration).
- `grid_search.py` for broader hyper-parameter sweeps.

### Grid Search & Model Selection

`notebooks/model_selection.ipynb` orchestrates:
- Cross-CFG sweeps (embedding CFG × image CFG).
- LAION aesthetic scoring using batched image evaluation.
- CVPR-ready heatmaps (posterior mean/median, LAION mean/median, Δ prior-to-posterior).

Artifacts are exported to `notebooks/outputs/cvpr_heatmaps/`.

---

## Evaluation & Diagnostics

- **`evaluation.py` / `notebooks/evaluation.ipynb`**  
  Scores generated images with multiple aesthetic predictors (HPSv2, LAION).  
  Provides per-user aggregates and persona-level breakdowns.

- **Bootstrap utilities** (see `notebooks/ncf_diagnostics.ipynb`) for confidence intervals and ROC analysis.

- **Visualization suite** (`notebooks/model_selection.ipynb`, `notebooks/viz_out/`) generates ECDFs, violin plots, threshold bars, pairwise deltas, and similarity matrices.

---

## Ablations & Reproducibility

`ablations.py` reproduces the primary ablation studies:

```bash
# Cross CFG sweep (embedding CFG × image CFG)
python ablations.py --experiment_type cross-cfgs

# Prompt ablation across baseline prompts
python ablations.py --experiment_type cross-prompts

# Generate baseline prompt-only samples
python ablations.py --experiment_type baseline-prompts \
  --prompt_level 2 --n_images 512 --dst_dir ./outputs/baselines/
```

Outputs are saved under `data/flickr/evaluation/ablations/` and consumed by the analysis notebooks.

---

## Notebooks

| Notebook | Purpose |
| --- | --- |
| `notebooks/model_selection.ipynb` | Cross-CFG analysis, LAION scoring, CVPR heatmaps |
| `notebooks/ncf_diagnostics.ipynb` | NCF training diagnostics, ROC & bootstrap AUC |
| `notebooks/evaluation.ipynb` | Persona scoring with HPSv2, LAION |
| `notebooks/permutation_test.ipynb` | Permutation tests for posterior medians |
| `notebooks/pp_viz/` | Posterior vs. prior visualizations (scatter, Bland–Altman, etc.) |

> All notebooks assume the processed data is in place and that the required models are available locally (diffusion weights, NCF checkpoints).

---

## Results & Figures

- Primary figures are stored in `notebooks/outputs/cvpr_heatmaps/` and `notebooks/outputs/model_selection/`.
- Additional supporting visuals (ECDFs, violin plots, rank stability) live in `notebooks/viz_out/`.
- Figures are exported as both PNG and PDF for camera-ready integration.

---

## Citation

If you use REBECA in your research, please cite:

```
@inprogress{rebeca2025,
  title   = {Recommendations Beyond Catalogs: Diffusion Models for Personalized Generation},
  author  = {Author, A. and Collaborator, B.},
  year    = {2025},
  note    = {Under review}
}
```

---

## License

This project is licensed under the [Apache License 2.0](LICENSE).

---

For questions or collaboration inquiries, please open an issue or reach out via GitHub Discussions. Let’s shape the future of generative recommenders together.
