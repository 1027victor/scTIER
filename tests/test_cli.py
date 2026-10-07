"""Exercise the installed dispatcher and actual analysis outputs."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def run(*args, ok=True):
    result = subprocess.run(
        [sys.executable, "-m", "sctier", *map(str, args)],
        capture_output=True, text=True, timeout=120,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "1"},
    )
    if ok:
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode != 0
    return result


@pytest.mark.parametrize("args, expected", [
    (("--version",), "scTIER 0.1.0"),
    (("--help",), "scGIFT"),
    (("scASprofiler", "--help"), "quantify"),
    (("scASprofiler", "perp", "--help"), "run"),
    (("scASprofiler", "perp", "run", "--help"), "--sj-dir"),
    (("scASprofiler", "impute", "--help"), "train"),
    (("scASprofiler", "impute", "train", "--help"), "--n-epochs"),
    (("scASprofiler", "quantify", "--help"), "--input-file"),
    (("scGIFT", "--help"), "--gene_counts"),
])
def test_help_is_forwarded(args, expected):
    assert expected in run(*args).stdout


def test_invalid_module():
    assert "invalid choice" in run("unknown", ok=False).stderr


def test_quantify(tmp_path):
    frame = pd.DataFrame({
        "coord.intron": ["j1", "j2", "j3"],
        "intron_group": ["g1", "g1", "g2"],
        "annotated": [1, 1, 1],
        "cell1": [2., 6., 3.], "cell2": [6., 2., 9.],
    })
    input_file = tmp_path / "counts.csv"
    frame.to_csv(input_file)
    run("scASprofiler", "quantify", "--input-file", input_file, "--outdir", tmp_path / "out")
    output = pd.read_csv(tmp_path / "out/as_ratio.csv", index_col=0)
    np.testing.assert_allclose(output["cell1"], [.25, .75, 1.])
    np.testing.assert_allclose(output["cell2"], [.75, .25, 1.])


def gift_inputs(tmp_path):
    np.save(tmp_path / "gene.npy", np.array([[2, 4], [4, 2], [3, 3], [1, 5]], dtype=np.float32))
    np.save(tmp_path / "iso.npy", np.array([[1, 1, 4], [3, 1, 2], [2, 1, 3], [1, 0, 5]], dtype=np.float32))
    np.save(tmp_path / "map.npy", np.array([0, 0, 1], dtype=np.int64))
    return ["scGIFT", "--gene_counts", tmp_path / "gene.npy", "--iso_counts", tmp_path / "iso.npy",
            "--iso2gene", tmp_path / "map.npy", "--epochs", "2", "--hid", "8", "--n_topics", "2",
            "--batch_size", "2", "--device", "cpu"]


def test_gift_training_and_repeatability(tmp_path):
    args = gift_inputs(tmp_path)
    for name in ("out1", "out2"):
        run(*args, "--outdir", tmp_path / name)
        theta = np.load(tmp_path / name / "theta.npy")
        mu = np.load(tmp_path / name / "embedding.npy")
        assert theta.shape == mu.shape == (4, 2)
        assert np.isfinite(theta).all() and np.isfinite(mu).all()
        np.testing.assert_allclose(theta.sum(1), 1., atol=1e-6)
        assert (tmp_path / name / "model.pt").is_file()
    np.testing.assert_array_equal(np.load(tmp_path / "out1/theta.npy"), np.load(tmp_path / "out2/theta.npy"))


@pytest.mark.parametrize("bad, expected", [
    (np.array([0., 0.5, 1.]), "integer gene indices"),
    (np.array([0, 0, 2]), "indices must be"),
])
def test_gift_rejects_invalid_mapping(tmp_path, bad, expected):
    args = gift_inputs(tmp_path)
    np.save(tmp_path / "map.npy", bad)
    assert expected in run(*args, ok=False).stderr


def test_gift_rejects_zero_epochs(tmp_path):
    assert "must be positive" in run(*gift_inputs(tmp_path), "--epochs", "0", ok=False).stderr


@pytest.mark.parametrize("tenx", [False, True])
def test_star_preprocessing(tmp_path, tenx):
    sj_dir = tmp_path / "sj"
    sj_dir.mkdir()
    # Two donors share one acceptor, both belonging to the same gene.
    for cell in range(3):
        (sj_dir / f"cell{cell}.SJ.out.tab").write_text(
            "chr1\t101\t200\t1\t1\t1\t10\t0\t25\n"
            "chr1\t121\t200\t1\t1\t1\t20\t0\t25\n"
        )
    gtf = tmp_path / "genes.gtf"
    gtf.write_text(
        'chr1\ttest\texon\t1\t100\t.\t+\t.\tgene_id "g1"; gene_name "G1";\n'
        'chr1\ttest\texon\t110\t120\t.\t+\t.\tgene_id "g1"; gene_name "G1";\n'
        'chr1\ttest\texon\t201\t300\t.\t+\t.\tgene_id "g1"; gene_name "G1";\n'
    )
    args = ["scASprofiler", "perp", "run", "--sj-dir", sj_dir, "--gtf", gtf,
            "--outdir", tmp_path / "out", "--samples-ps", "1", "--sites-ps", "1",
            "--sites-thres", "0", "--samples-thres", "0"]
    if tenx:
        args.append("--x10")
    run(*args)
    matrix = pd.read_csv(tmp_path / "out/filter_sj_counts.csv", sep="\t", index_col=0)
    assert matrix.shape[0] == 2
    assert list(matrix.columns[-3:]) == ["intron_group", "annotated", "coord.intron"]
    assert set(matrix.columns[:3]) == {"cell0", "cell1", "cell2"}
    np.testing.assert_allclose(matrix.iloc[:, :3].values, [[10., 10., 10.], [20., 20., 20.]])


def test_impute_train_and_fill(tmp_path):
    rng = np.random.default_rng(42)
    frame = pd.DataFrame(rng.integers(1, 20, size=(16, 4)).astype(float),
                         columns=["cell0", "cell1", "cell2", "cell3"])
    frame.iloc[0, 0] = np.nan
    frame["intron_group"] = [f"g{i // 2}" for i in range(16)]
    frame["annotated"] = 1
    frame["coord.intron"] = [f"j{i}" for i in range(16)]
    counts = tmp_path / "counts.tsv"
    frame.to_csv(counts, sep="\t")
    labels = tmp_path / "labels.csv"
    labels.write_text("A\nA\nB\nB\n")
    output = tmp_path / "out"
    run("scASprofiler", "impute", "train", "--data-Sj", counts, "--data-c", labels,
        "--outdir", output, "--name", "smoke", "--n-epochs", "1", "--batch-size", "4",
        "--clusters", "2", "--latent-dim", "4", "--drop-prob", "0.25",
        "--run-impute", "--sim-size", "4", "-k", "2")
    for kind in ("enc", "dec", "d"):
        assert (output / "models" / f"smoke-4-1-2-{kind}_best.pt").is_file()
    filled = pd.read_csv(output / "scASP-smoke.csv", index_col=0)
    assert filled.shape == (16, 7)
    assert np.isfinite(filled[["cell0", "cell1", "cell2", "cell3"]].values).all()
