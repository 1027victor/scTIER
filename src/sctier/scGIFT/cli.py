"""scGIFT command line interface (lightweight help, lazy model imports)."""
import argparse
import math


def build_parser():
    ap = argparse.ArgumentParser(prog="scGIFT", description="Isoform+Gene Neural Topic Fusion (VAE)")
    ap.add_argument("--gene_counts", type=str, required=True, help="gene_counts.npy (cells x genes)")
    ap.add_argument("--iso_counts", type=str, required=True, help="iso_counts.npy (cells x isoforms)")
    ap.add_argument("--iso2gene", type=str, required=True, help="iso2gene.npy (isoforms,) -> gene index")
    ap.add_argument("--outdir", type=str, default="out_isotopic", help="output directory")
    ap.add_argument("--n_topics", type=int, default=30)
    ap.add_argument("--hid", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--alpha_iso", type=float, default=1.0, help="weight for isoform likelihood")
    ap.add_argument("--beta_kl_max", type=float, default=1)
    ap.add_argument("--kl_anneal_epochs", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    return ap


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    for name in ("n_topics", "hid", "epochs", "batch_size", "kl_anneal_epochs"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be finite and positive")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    for name in ("alpha_iso", "beta_kl_max"):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f"--{name} must be finite and nonnegative")
    if not 0 <= args.seed < 2**32:
        parser.error("--seed must be in [0, 2**32)")
    from .iso_topic_fusion import run
    try:
        run(args)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
