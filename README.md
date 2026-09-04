# 🚀 SCOUT: Structure-Aware Aspect and Anchor-Count Selection for Node Attribute Augmentation via Positional Information

[![Paper - WWW 2026](https://img.shields.io/badge/WWW%202026-Accepted-blue.svg)](https://doi.org/10.1145/3774904.3792326)
[![DOI](https://img.shields.io/badge/DOI-10.1145%2F3774904.3792326-blue.svg)](https://doi.org/10.1145/3774904.3792326)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-≥3.8-blue.svg)]()
[![Framework](https://img.shields.io/badge/framework-PyTorch%20%7C%20PyG-orange)]()

**SCOUT** is a **model-agnostic augmentation framework** that enhances graph neural
networks (GNNs) when node attributes are **missing**, **sparse**, or
**uninformative**, by leveraging **multi-aspect positional information (PI)** and a
**graph-aware anchor-selection mechanism**.

---

## 📖 Abstract

> When node attributes are absent or limited, GNNs often fail to distinguish
> structurally similar nodes, leading to degraded downstream performance.
> Positional-information augmentation addresses this by selecting representative
> nodes as **anchors** and encoding each node's relation to those anchors as new
> features. Its effectiveness, however, hinges on two graph-dependent choices:
> (1) the **structural measures** used for anchor selection and node–anchor
> scoring, and (2) the **anchor-count `K`**.
>
> **SCOUT** removes the manual tuning by:
> - Selecting **positional aspects** (centrality–similarity pairs) via a
>   **graph-level attention** selector.
> - Determining the **anchor-count per graph** with a principled **elbow method**
>   grounded in power-law centrality distributions.
> - Augmenting node features with positional information that complements the
>   original attributes when they are present.
>
> This yields consistent gains on **link prediction** and **node classification**,
> both with and without original node attributes.

📄 **Paper**: *SCOUT: Structure-Aware Aspect and Anchor-Count Selection for Node
Attribute Augmentation via Positional Information*
🌐 **Venue**: [The Web Conference (WWW) 2026](https://www2026.thewebconf.org) ·
[doi.org/10.1145/3774904.3792326](https://doi.org/10.1145/3774904.3792326)
📁 **Code**: <https://github.com/seinkim01/SCOUT>

---

## ✅ What is in this repository

This is a **reference implementation** of the SCOUT pipeline. It is scoped to the
citation-graph experiments and the analyses behind the paper's method section.

| Component | Status | Entry point |
|---|---|---|
| Positional attribute generation (16 aspect blocks) | ✅ implemented | `src/core/generate_attributes.py` |
| Elbow-based anchor-count selection | ✅ implemented | `src/core/elbow_selector.py` |
| Graph-level aspect attention gate | ✅ implemented | `src/models/attr_gate.py` |
| **Link prediction** (Planetoid: Cora / Citeseer / Pubmed) | ✅ implemented | `src/core/train_linkpred.py` |
| **Node classification** (Planetoid: Cora / Citeseer / Pubmed) | ✅ implemented | `src/core/train_nodeclf.py` |
| Elbow / power-law diagnostics & K-sweep | ✅ implemented | `src/analysis/` |
| OGB benchmarks (`ogbl-ddi`, `ogbn-arxiv`, `ogbn-mag`) | ⚠️ generation code paths present; training/eval harness **not included here** — see the paper |
| Baseline methods (HPLC, P-GNN, SEAL, …) | ⚠️ compared **in the paper**; not vendored in this repo |

> Amazon (Computers / Photo) and Coauthor-CS also have data-loading paths and can
> be run with the same scripts, but were not part of the reported main tables.

---

## ⚙️ Installation

```bash
# Recommended: Conda
conda env create -f requirements_conda.yaml
conda activate scout

# Or: pip (into a fresh Python ≥3.8 environment)
pip install -r requirements.txt
```

The pinned stack is **PyTorch 1.13 + CUDA 11.7 + PyG 2.6**. Newer PyTorch/PyG
combinations also run the citation-graph experiments.

---

## 📦 Data preparation

- **Graphs.** PyTorch Geometric downloads Planetoid automatically into
  `datasets/<Name>/` on first use. A copy of the **Cora** raw files is committed
  so the quick-start works offline.
- **Positional attributes.** A ready-made Cora example is committed under
  `attrs/Cora_concat_centrality/` (`concat_all_top10.684.npy` +
  `meta_concat_all_top10.684.json`). Regenerate it, or build other datasets, with:

```bash
python -m src.core.generate_attributes \
  --dataset Cora --data_root ./datasets \
  --output_dir ./attrs/Cora_concat_centrality --prefix concat_all
# omit --top_k_percent to let the elbow selector choose K per graph
```

This produces the concatenated **16-block** aspect tensor
(`{betweenness, closeness, eigenvector, pagerank} × {RWR, AdaSim, RA, Jaccard}`)
and a meta JSON describing `block_dims`, the chosen anchors, and the elbow
metadata. Attributes are built from the **train split only** (leakage-safe).

---

## 🚀 Quick start

```bash
# Link prediction on Cora (w/o and w/ original features)
bash scripts/run_linkpred.sh

# Node classification on Cora (w/o and w/ original features)
bash scripts/run_nodeclf.sh
```

Direct invocation:

```bash
python -m src.core.train_linkpred \
  --dataset Cora --data_root ./datasets \
  --attr_file ./attrs/Cora_concat_centrality/concat_all_top10.684.npy \
  --meta_file ./attrs/Cora_concat_centrality/meta_concat_all_top10.684.json \
  --model gcn --decoder mlp --hidden 64 --layer 2 --epochs 2000
# add --use_raw_feature to concatenate the original node features

python -m src.core.train_nodeclf \
  --dataset Cora --data_root ./datasets \
  --attr_file ./attrs/Cora_concat_centrality/concat_all_top10.684.npy \
  --meta_file ./attrs/Cora_concat_centrality/meta_concat_all_top10.684.json \
  --model gcn --hidden 64 --layer 2 --epochs 1000 --patience 20
```

Reference flag sets are in [`configs/`](./configs).

---

## 🔬 Reproducing the paper's analyses

```bash
# Elbow points + power-law fits + ranked/log-log centrality curves
bash scripts/run_c3_elbow_analysis.sh                 # -> results/c3_elbow/

# Anchor-count K sweep vs. elbow K
bash scripts/run_c3_k_sweep.sh                        # -> results/c3_k_sweep/

# Figures used in the paper (elbow visualisation experiments)
bash scripts/run_elbow_visualization_experiments.sh   # -> figures/
```

The committed `figures/*.{png,pdf}` and `figures/results_elbow_analysis.csv` are
the outputs of the last command.

---

## 🧱 Project structure

```
SCOUT/
├── attrs/
│   └── Cora_concat_centrality/     # committed Cora example (16-block aspect tensor + meta)
├── configs/                        # reference hyper-parameters per shipped experiment
├── datasets/Cora/raw/              # committed Cora raw files (offline example)
├── figures/                        # elbow-analysis figures + CSV (paper artifacts)
├── scripts/                        # runnable wrappers for training + analyses
├── src/
│   ├── core/
│   │   ├── generate_attributes.py  # leakage-safe positional attribute generator
│   │   ├── elbow_selector.py       # power-law / Kneedle elbow anchor-count selector
│   │   ├── train_linkpred.py       # link-prediction trainer (AUC / AP / MRR)
│   │   └── train_nodeclf.py        # node-classification trainer (Acc / macro-F1)
│   ├── models/
│   │   ├── attr_gate.py            # MeasureAttentionGateV3 — graph-level aspect attention
│   │   ├── encoder.py              # GCN / GraphSAGE encoder
│   │   ├── decoder.py              # inner-product / MLP link decoders
│   │   └── heads.py                # MLP node-classification head
│   ├── utils/data_loader.py        # link-pred & node-clf dataset loaders
│   └── analysis/                   # elbow / power-law diagnostics, K-sweep summaries
├── requirements.txt · requirements_conda.yaml
├── CITATION.cff · LICENSE
```

---

## 📊 Results

The numbers below are **as reported in the paper**
([doi.org/10.1145/3774904.3792326](https://doi.org/10.1145/3774904.3792326));
see the paper for full tables, baselines, and standard deviations.

| Setting | Task | Dataset | Reported gain |
|---|---|---|---|
| w/o original attributes | Link prediction | `ogbl-ddi` | **+26.88% Hits@20** |
| w/o original attributes | Node classification | `ogbn-arxiv` | **+4.52% accuracy** |
| w/ original attributes | Node classification | `ogbn-mag` | **+11.69% accuracy** |

To reproduce the shipped Cora link-prediction run locally:

```bash
bash scripts/run_linkpred.sh   # writes logs/cora_wofeat_gcn.log and logs/cora_wfeat_gcn.log
```

---

## 🛠️ Troubleshooting

- **`Attribute files not found`** — pass `--attr_file` / `--meta_file` that match
  a folder produced by `generate_attributes.py` (the `meta_*.json` must sit next
  to its `*.npy`).
- **CUDA out of memory** — reduce `--hidden`, `--att_dim`, or the anchor count
  (`--max_anchors` / a smaller `--top_k_percent`) when generating attributes.
- **Wrong GPU** — the scripts set `CUDA_VISIBLE_DEVICES=0`; override in the
  environment.

---

## 👥 Contributors

- **Dong-Hyuk Seo** — Hanyang University
- **Sein Kim** — Hanyang University
- **Taeri Kim** — Hanyang University
- **Won-Yong Shin** — Yonsei University
- **Sang-Wook Kim** — Hanyang University

---

## 📄 Citation

```bibtex
@inproceedings{seo2026scout,
  title     = {SCOUT: Structure-Aware Aspect and Anchor-Count Selection for
               Node Attribute Augmentation via Positional Information},
  author    = {Seo, Dong-Hyuk and Kim, Sein and Kim, Taeri and
               Shin, Won-Yong and Kim, Sang-Wook},
  booktitle = {Proceedings of the ACM Web Conference 2026 (WWW '26)},
  year      = {2026},
  publisher = {ACM},
  doi       = {10.1145/3774904.3792326}
}
```

## 📝 License

Released under the [MIT License](./LICENSE).
