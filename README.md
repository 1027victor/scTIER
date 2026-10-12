<h1 align="center">scTIER</h1>

<p align="center">
  <strong>a computational framework for multi-level transcriptomic analysis in single cells</strong>
</p>

<p align="center">
  <a href="https://github.com/1027victor/scTIER/actions/workflows/tests.yml"><img src="https://github.com/1027victor/scTIER/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Python-%E2%89%A53.10.9-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10.9 or later"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-12866E?style=flat-square" alt="Apache 2.0 license"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#scasprofiler">scASprofiler</a> ·
  <a href="#scgift">scGIFT</a> ·
  <a href="#resources">Resources</a>
</p>

scTIER combines **scASprofiler** for alternative splicing analysis and **scGIFT** for gene–isoform integration. Both modules are installed together and run through their own commands.

## Installation

Requires **Python ≥ 3.10.9**. Linux is the primary development platform.

```bash
conda create -n sctier python=3.10 -y
conda activate sctier
pip install git+https://github.com/1027victor/scTIER.git
```

For GPU use, install a compatible [PyTorch build](https://pytorch.org/get-started/locally/) before installing scTIER.

<details>
<summary><b>Local installation and optional Ray support</b></summary>

```bash
git clone https://github.com/1027victor/scTIER.git
cd scTIER
pip install .
```

To enable Ray-based parallel preprocessing with `--use-ray`:

```bash
pip install 'scTIER[ray] @ git+https://github.com/1027victor/scTIER.git'
```

</details>

## Quick start

Use the module commands directly:

```bash
scASprofiler perp run --help
scASprofiler impute train --help
scASprofiler quantify --help
scGIFT --help
```

<details>
<summary><b>Try scGIFT with a small synthetic dataset</b></summary>

Generate count matrices for 12 cells, 3 genes, and 6 isoforms:

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
  --n_topics 30 --hid 512 --epochs 200 --batch_size 256 \
  --device auto --outdir out_isotopic
```

The example uses the default training settings and saves outputs in `out_isotopic/`.

</details>

## scASprofiler

Preprocess splice junctions, impute missing counts with a VAE-GAN and class-conditional KNN refinement, and quantify alternative splicing.

### 1. Preprocess

Place STAR `.SJ.out.tab` files directly in `SJ_files/`, with a unique filename for each cell or sample, and supply a GTF annotation.

```bash
scASprofiler perp run \
  --sj-dir ./SJ_files \
  --gtf ./annotation.gtf \
  --outdir ./out_perp
```

Plate-based processing is the default; add `--x10` for 10x data. Adjust `--samples-ps`, `--sites-ps`, `--sites-thres`, and `--samples-thres` for the dataset's size and coverage.

| Output | Contents |
| --- | --- |
| `merged_sj.tab` | Merged splice junction records. |
| `raw_sj_counts.csv` | Junction-by-cell count matrix. |
| `sj_meta.csv` | Filtered splice junction metadata. |
| `filter_sj_counts.csv` | Filtered counts and metadata for imputation. |

All four preprocessing outputs are **tab-separated**, including files with a `.csv` extension.

### 2. Train and impute

| Input | Format |
| --- | --- |
| `--data-Sj` | Tab-separated junction-by-cell matrix with a row index. The final three columns are `intron_group`, `annotated`, and `coord.intron`. Missing counts are NaN; each cell needs at least one positive count for normalization. |
| `--data-c` | One-column CSV with no header or index, containing one label per cell in the same order as the matrix's cell columns. String or numeric labels are encoded automatically. |

```bash
scASprofiler impute train \
  --data-Sj ./out_perp/filter_sj_counts.csv \
  --data-c ./cell_labels.csv \
  --outdir ./out_impute --name example \
  --clusters 4 --latent-dim 100 --n-epochs 200 \
  --run-impute
```

Set `--clusters` to at least the number of label categories. This example saves model checkpoints in `out_impute/models/` and imputed counts in `out_impute/scASP-example.csv`.

<details>
<summary><b>Impute with a trained decoder</b></summary>

```bash
scASprofiler impute impute \
  --data-Sj ./out_perp/filter_sj_counts.csv \
  --data-c ./cell_labels.csv \
  --outdir ./out_impute --name example \
  --clusters 4 --latent-dim 100 \
  --decoder-ckpt ./out_impute/models/example-100-200-4-dec_best.pt
```

Reuse the training matrix and label file, and keep `--clusters`, `--latent-dim`, and `--channels` consistent with training.

</details>

### 3. Quantify

```bash
scASprofiler quantify \
  --input-file ./out_impute/scASP-example.csv \
  --outdir ./out_as
```

Writes `out_as/as_ratio.csv`, with junction counts normalized by their intron-group totals within each cell.

## scGIFT

Integrate gene and isoform counts with a two-view neural topic model to obtain cell embeddings and topic proportions.

### Inputs

| Argument | NumPy `.npy` array |
| --- | --- |
| `--gene_counts` | `(n_cells, n_genes)` — finite, nonnegative gene counts. |
| `--iso_counts` | `(n_cells, n_isoforms)` — finite, nonnegative isoform counts. |
| `--iso2gene` | `(n_isoforms,)` — integer gene-column indices from `0` to `n_genes - 1`. |

Both count matrices must use the same cell order. The isoform columns must follow the order of `iso2gene`.

### Run

```bash
scGIFT \
  --gene_counts ./gene_counts.npy \
  --iso_counts ./iso_counts.npy \
  --iso2gene ./iso2gene.npy \
  --n_topics 20 --hid 128 --epochs 300 --batch_size 64 \
  --alpha_iso 1.0 --seed 42 --outdir ./out_scgift
```

CUDA is selected automatically when available. Use `--device cpu` or `--device cuda` to select a device explicitly.

<details>
<summary><b>Training options and defaults</b></summary>

| Option | Default | Purpose |
| --- | --- | --- |
| `--n_topics` | `30` | Topics and embedding dimensions. |
| `--hid` | `512` | Encoder hidden-layer width. |
| `--epochs` | `200` | Training epochs. |
| `--batch_size` | `256` | Cells per batch. |
| `--lr` | `0.002` | Learning rate. |
| `--alpha_iso` | `1.0` | Isoform likelihood weight. |
| `--beta_kl_max` | `1.0` | Maximum KL regularization weight. |
| `--kl_anneal_epochs` | `50` | KL warm-up epochs. |
| `--dropout` | `0.1` | Encoder dropout probability. |
| `--seed` | `42` | Random seed. |
| `--device` | `auto` | `auto`, `cpu`, or `cuda`. |

</details>

### Outputs

| File | Contents |
| --- | --- |
| `embedding.npy` | Posterior means, `(n_cells, n_topics)`. |
| `theta.npy` | Topic proportions, `(n_cells, n_topics)`; each row sums to one. |
| `model.pt` | PyTorch state dictionary, including the isoform-to-gene mapping. |

Output rows preserve the input cell order. Keep the model configuration and cell, gene, and isoform identifiers with the checkpoint.

## Resources

[scASprofiler documentation](https://scasprofiler.readthedocs.io/) · [Report an issue](https://github.com/1027victor/scTIER/issues) · [Source provenance](UPSTREAM.md)

<details>
<summary><b>Development</b></summary>

From a local clone:

```bash
pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' pytest -q
python -m build
```

</details>

The bundled scASprofiler module is based on [1027victor/scASprofiler](https://github.com/1027victor/scASprofiler). scTIER is licensed under [Apache 2.0](LICENSE).
