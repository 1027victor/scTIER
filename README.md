# scTIER

**a computational framework for multi-level transcriptomic analysis in single cells**

[![Tests](https://github.com/1027victor/scTIER/actions/workflows/tests.yml/badge.svg)](https://github.com/1027victor/scTIER/actions/workflows/tests.yml)
[![Python](https://img.shields.io/badge/python-%3E%3D3.10.9-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)

scTIER brings two analysis modules into one installable Python package:

| Module | Purpose | Command |
| --- | --- | --- |
| **scASprofiler** | Preprocess STAR splice junction counts, impute missing counts with a VAE-GAN and class-conditional KNN refinement, and calculate alternative splicing ratios. | `scASprofiler` |
| **scGIFT** | Learn a shared cell representation from gene and isoform counts using a two-view neural topic model with within-gene isoform constraints. | `scGIFT` |

Install scTIER once, then run either module directly from the command line. Both modules are included in the package and can be used independently.

[Installation](#installation) · [Quick start](#quick-start) · [scASprofiler](#scasprofiler) · [scGIFT](#scgift) · [Development](#development) · [Support](#support-and-contributing)

## Installation

Requires **Python 3.10.9 or later**. Linux is the primary development and validation platform. A dedicated environment avoids conflicts with an existing scASprofiler installation.

```bash
conda create -n sctier python=3.10 -y
conda activate sctier
pip install git+https://github.com/1027victor/scTIER.git
```

Alternatively, install from a local clone:

```bash
git clone https://github.com/1027victor/scTIER.git
cd scTIER
pip install .
```

The installation includes Click, NumPy, pandas, PyTorch, and tqdm. For GPU execution, install a compatible PyTorch build using the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) before installing scTIER.

For optional Ray-based parallel preprocessing:

```bash
pip install 'scTIER[ray] @ git+https://github.com/1027victor/scTIER.git'
```

STAR and Cell Ranger are separate upstream tools. The scASprofiler preprocessing command reads existing STAR `SJ.out.tab` files; it does not perform read alignment.

## Quick start

### Explore the command-line interface

```bash
scASprofiler perp run --help
scASprofiler impute train --help
scASprofiler quantify --help
scGIFT --help
```

**scTIER is the installation package name. Use `scASprofiler` and `scGIFT` directly for analysis, without a `scTIER` prefix.**

### Run scGIFT on a small example

Create aligned gene and isoform count matrices and an isoform-to-gene mapping:

```bash
python - <<'PY'
from pathlib import Path
import numpy as np

out = Path("demo")
out.mkdir(exist_ok=True)
rng = np.random.default_rng(42)
iso_counts = rng.poisson(3, size=(12, 6)).astype(np.float32)
iso2gene = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
gene_counts = iso_counts.reshape(12, 3, 2).sum(axis=2)

np.save(out / "gene_counts.npy", gene_counts)
np.save(out / "iso_counts.npy", iso_counts)
np.save(out / "iso2gene.npy", iso2gene)
PY

scGIFT \
  --gene_counts demo/gene_counts.npy \
  --iso_counts demo/iso_counts.npy \
  --iso2gene demo/iso2gene.npy \
  --n_topics 3 --hid 16 --epochs 2 --batch_size 6 \
  --device cpu --outdir demo/results
```

This example verifies that the analysis runs and writes its outputs. Use appropriate model settings and training duration for your research dataset.

## scASprofiler

scASprofiler provides preprocessing, imputation, and AS ratio calculation. Detailed methods and upstream tutorials are available in the [scASprofiler documentation](https://scasprofiler.readthedocs.io/).

### 1. Preprocess splice junction counts

Place the input `.SJ.out.tab` files directly inside `SJ_files/`, with unique filenames for each sample or cell, and supply a GTF annotation:

```bash
scASprofiler perp run \
  --sj-dir ./SJ_files \
  --gtf ./annotation.gtf \
  --outdir ./out_perp
```

The default pipeline is for plate-based data. Add `--x10` to use the 10x pipeline. Adjust `--samples-ps`, `--sites-ps`, `--sites-thres`, and `--samples-thres` to suit the dataset size and coverage. Add `--use-ray` to enable parallel threshold filtering after installing the Ray extra.

| Output | Contents |
| --- | --- |
| `merged_sj.tab` | Merged splice junction records. |
| `raw_sj_counts.csv` | Junction-by-cell count matrix, stored as a tab-separated file. |
| `sj_meta.csv` | Filtered splice junction metadata, stored as a tab-separated file. |
| `filter_sj_counts.csv` | Filtered count matrix with metadata columns, stored as a tab-separated file. |

### 2. Train and impute

The imputation module expects:

- **`--data-Sj`**: a tab-separated junction-by-cell matrix with a row index. The final three columns must be `intron_group`, `annotated`, and `coord.intron`. Missing counts are represented as NaN; observed zeros remain observed values.
- **`--data-c`**: a single-column CSV with no header or index, containing one label per cell. Labels may be strings or numbers and are encoded automatically. Their row order must match the cell columns of the count matrix.
- **`--clusters`**: at least the number of distinct cell labels. Use matching cluster and latent-dimension settings when reusing a checkpoint.

Train the model and immediately run imputation:

```bash
scASprofiler impute train \
  --data-Sj ./out_perp/filter_sj_counts.csv \
  --data-c ./cell_labels.csv \
  --outdir ./out_impute --name example \
  --clusters 4 --latent-dim 100 --n-epochs 200 \
  --run-impute
```

The example assumes four cell-label categories. Adjust `--clusters` for your input labels. The command saves the best encoder, decoder, and discriminator checkpoints under `out_impute/models/` and writes the imputed counts to `out_impute/scASP-example.csv`.

To impute with the saved decoder:

```bash
scASprofiler impute impute \
  --data-Sj ./out_perp/filter_sj_counts.csv \
  --data-c ./cell_labels.csv \
  --outdir ./out_impute --name example \
  --clusters 4 --latent-dim 100 \
  --decoder-ckpt ./out_impute/models/example-100-200-4-dec_best.pt
```

### 3. Calculate alternative splicing ratios

```bash
scASprofiler quantify \
  --input-file ./out_impute/scASP-example.csv \
  --outdir ./out_as
```

The command normalizes junction counts by their intron-group totals within each cell and writes `out_as/as_ratio.csv`.

## scGIFT

scGIFT jointly models gene expression and isoform usage to produce cell embeddings and topic proportions. Its gene decoder uses a softmax over genes, while its isoform decoder normalizes isoforms within their mapped gene.

### Input data

Provide three NumPy `.npy` files:

| Argument | Shape | Requirements |
| --- | --- | --- |
| `--gene_counts` | `(n_cells, n_genes)` | Finite, nonnegative gene counts. |
| `--iso_counts` | `(n_cells, n_isoforms)` | Finite, nonnegative isoform counts, with the same cell order as the gene matrix. |
| `--iso2gene` | `(n_isoforms,)` | Integer mapping from each isoform to a gene column, using indices from `0` to `n_genes - 1`. |

The isoform column order must match `iso2gene`. These are gene and isoform count inputs; the scASprofiler junction matrix is not an automatic substitute for isoform counts.

### Train the fusion model

```bash
scGIFT \
  --gene_counts ./gene_counts.npy \
  --iso_counts ./iso_counts.npy \
  --iso2gene ./iso2gene.npy \
  --n_topics 20 --hid 128 --epochs 300 --batch_size 64 \
  --alpha_iso 1.0 --seed 42 --outdir ./out_scgift
```

| Option | Default | Description |
| --- | --- | --- |
| `--n_topics` | `30` | Number of topics and embedding dimensions. |
| `--hid` | `512` | Encoder hidden-layer width. |
| `--epochs` | `200` | Number of training epochs. |
| `--batch_size` | `256` | Cells per batch. |
| `--lr` | `0.002` | Learning rate. |
| `--alpha_iso` | `1.0` | Weight of the isoform likelihood. |
| `--beta_kl_max` | `1.0` | Maximum KL regularization weight. |
| `--kl_anneal_epochs` | `50` | KL warm-up duration. |
| `--dropout` | `0.1` | Encoder dropout probability. |
| `--seed` | `42` | Random seed. |
| `--device` | `auto` | Select `auto`, `cpu`, or `cuda`. |

By default, scGIFT uses CUDA when available and otherwise uses the CPU. It enables deterministic PyTorch algorithms and disables TF32. See `scGIFT --help` for the complete interface.

### Outputs

| File | Contents |
| --- | --- |
| `embedding.npy` | Posterior means with shape `(n_cells, n_topics)`. |
| `theta.npy` | Topic proportions with shape `(n_cells, n_topics)`; each row sums to one. |
| `model.pt` | PyTorch model state dictionary. |

Output rows preserve the input cell order. Keep the training command, gene and isoform vocabularies, and `iso2gene` mapping alongside the checkpoint: the state dictionary alone does not store all information needed to reconstruct the model.

## Development

```bash
git clone https://github.com/1027victor/scTIER.git
cd scTIER
pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' pytest -q
python -m build
```

The tests cover CLI argument forwarding, STAR preprocessing, AS ratio calculation, VAE-GAN training and imputation, scGIFT training on small count matrices, and input validation. GitHub Actions runs tests and package builds on Python 3.10 and 3.12.

## Support and contributing

For questions, bug reports, or feature requests, open a [GitHub issue](https://github.com/1027victor/scTIER/issues). Include the command, package versions, error traceback, and a small reproducible example where possible. Pull requests for fixes, tests, and documentation improvements are welcome.

## Attribution and citation

The bundled scASprofiler code originates from [1027victor/scASprofiler](https://github.com/1027victor/scASprofiler). The scGIFT implementation originates from the author's gene–isoform topic fusion script. Source versions and packaging changes are recorded in [UPSTREAM.md](UPSTREAM.md).

When reporting analyses, include the scTIER repository URL and the release or Git commit used. Consult the upstream scASprofiler documentation for any method-specific citation instructions.

## License

scTIER is distributed under the [Apache License 2.0](LICENSE).
