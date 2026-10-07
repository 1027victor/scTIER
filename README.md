# scTIER

scTIER 将两个分析模块封装为一个 Python 安装包：

- **scASprofiler**：STAR 剪接位点预处理、VAE-GAN 插补、AS 比例计算，源码来自 [1027victor/scASprofiler](https://github.com/1027victor/scASprofiler)。
- **scGIFT**：基因表达与异构体计数的双视图神经主题融合，输出细胞嵌入、主题比例和模型参数。

两个模块随 scTIER 一起安装，不需要单独安装 scASprofiler 或拉取 Git submodule。Python 导入路径为 `sctier.scASprofiler` 和 `sctier.scGIFT`。原始数据和本地环境不包含在仓库中。

## 安装

建议在独立环境中使用 Python 3.10.9 或更新版本（已用 Python 3.10 验证），Linux 为优先使用平台。

```bash
conda create -n sctier python=3.10 -y
conda activate sctier
# GitHub 仓库发布后可以直接安装：
pip install git+https://github.com/1027victor/scTIER.git
```

本地源码安装：

```bash
cd scTIER
pip install .
# 开发安装：pip install -e '.[dev]'
```

默认安装会安装两个模块需要的 Python 依赖（包含 PyTorch）。GPU 用户可以先按 [PyTorch 安装说明](https://pytorch.org/get-started/locally/) 安装适合硬件的 PyTorch，再安装 scTIER。STAR/Cell Ranger 等上游比对软件需要单独安装；scASprofiler 预处理读取已生成的 `SJ.out.tab`。

使用 `--use-ray` 并行预处理时额外安装：

```bash
pip install '.[ray]'
# 或仓库发布后：pip install 'scTIER[ray] @ git+https://github.com/1027victor/scTIER.git'
```

## 命令入口

安装 scTIER 后，直接使用两个模块的命令：

```bash
scASprofiler perp run --help
scASprofiler impute train --help
scASprofiler quantify --help
scGIFT --help
```

scTIER 是安装包名称。日常执行直接使用 `scASprofiler` 和 `scGIFT`，无需添加 `scTIER` 前缀。若环境中已有独立 scASprofiler 安装，建议新建环境，避免同名命令覆盖。

## scASprofiler

保留上游参数和处理流程。详细输入格式、参数和教程见 [上游文档](https://scasprofiler.readthedocs.io/)。以下路径为示例，请替换为实际文件。

### 剪接位点预处理

```bash
scASprofiler perp run \
  --sj-dir ./SJ_files --gtf ./annotation.gtf --outdir ./out_perp
# 10x 数据在上述命令末尾加 --x10
```

输出 `merged_sj.tab`、`raw_sj_counts.csv`、`sj_meta.csv`、`filter_sj_counts.csv`。默认筛选阈值面向实际数据；小规模数据可按需调整 `--samples-ps`、`--sites-ps`、`--sites-thres`、`--samples-thres`。

### 训练与插补

```bash
scASprofiler impute train \
  --data-Sj ./out_perp/filter_sj_counts.csv \
  --data-c ./cell_labels.csv \
  --outdir ./out_impute --n-epochs 200 --run-impute

scASprofiler impute impute \
  --data-Sj ./out_perp/filter_sj_counts.csv \
  --data-c ./cell_labels.csv \
  --outdir ./out_impute --decoder-ckpt ./out_impute/models/example-dec_best.pt
```

`--data-Sj` 使用制表符分隔，细胞为列，最后三列为 `intron_group`、`annotated`、`coord.intron`；`--data-c` 是无表头、无索引的一列 CSV，每行一个细胞标签（字符串或整数均可），上游自动编码为零起始整数；`--clusters` 必须不小于标签类别数。标签顺序必须与计数矩阵中的细胞顺序一致。插补输出文件名由上游按输入名/`--name` 与训练参数生成，具体路径以命令输出为准。

### AS 比例计算

```bash
scASprofiler quantify \
  --input-file ./imputed_junction_counts.csv --outdir ./out_as
```

读取插补结果 CSV，输出 `as_ratio.csv`。

## scGIFT

输入三个 `.npy` 文件：

| 参数 | 形状与含义 |
| --- | --- |
| `--gene_counts` | `(n_cells, n_genes)`，非负有限原始计数 |
| `--iso_counts` | `(n_cells, n_isoforms)`，非负有限原始计数，细胞顺序与基因矩阵一致 |
| `--iso2gene` | `(n_isoforms,)`，整数索引，第 i 个异构体对应基因矩阵的哪一列，从 0 开始 |

```bash
scGIFT \
  --gene_counts ./gene_counts.npy \
  --iso_counts ./iso_counts.npy \
  --iso2gene ./iso2gene.npy \
  --n_topics 20 --hid 128 --epochs 300 --batch_size 64 \
  --alpha_iso 1.0 --seed 42 --outdir ./out_scgift
```

默认自动选择 CUDA 或 CPU；`--device cpu` 强制 CPU，`--device cuda` 强制 CUDA。保持原脚本的确定性训练设置。

输出：`embedding.npy` 为后验均值 `(n_cells, n_topics)`，`theta.npy` 为主题比例（每行和为 1），`model.pt` 为 PyTorch state dict。输出细胞顺序与输入一致。保存模型参数不包含词表、索引映射或超参数，请同时保留这些输入和训练命令以便复用。

## 开发验证

```bash
pip install -e '.[dev]'
CUDA_VISIBLE_DEVICES='' pytest -q
python -m build
```

测试覆盖命令参数转发、AS 比例计算、STAR 预处理、VAE-GAN 训练和插补、scGIFT 小矩阵训练与输入校验。小数据验证不替代真实数据的科学效果评估。

源码来源和封装变更见 [UPSTREAM.md](UPSTREAM.md)。许可证为 Apache-2.0。

## 首次发布到 GitHub

本地项目准备好后，使用已获授权的 `1027victor` 账号创建公开仓库：

```bash
cd scTIER
gh auth login
gh repo create 1027victor/scTIER --public --source=. --remote=origin --push
```

这些命令需要 GitHub CLI 和仓库创建权限；GitHub 发布完成后，上面的远程 `pip install` 命令才可使用。本项目尚未发布到 PyPI，因此安装时请使用源码路径、wheel 或 GitHub 地址。
