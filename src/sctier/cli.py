"""Unified dispatcher; downstream parsers retain all original arguments."""
import argparse
import sys
from importlib import import_module

from . import __version__

AS_COMMANDS = {
    "perp": ("sctier.scASprofiler.scASp_perp.cli", "cli"),
    "impute": ("sctier.scASprofiler.scASp_impute.cli", "cli"),
    "quantify": ("sctier.scASprofiler.scASp_quantify.calculate_as_ratio", "main"),
}


def _invoke(action, argv):
    module, entry = AS_COMMANDS[action]
    command = getattr(import_module(module), entry)
    return command.main(args=argv, prog_name=f"scASprofiler-{action}")


def scasprofiler(argv=None):
    parser = argparse.ArgumentParser(prog="scASprofiler", description="Single-cell alternative splicing analysis")
    parser.add_argument("action", choices=AS_COMMANDS, help="perp: preprocessing; impute: training/imputation; quantify: AS ratios")
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser.parse_args(argv[:1])
    return _invoke(args.action, argv[1:])


def perp():
    return _invoke("perp", sys.argv[1:])


def impute():
    return _invoke("impute", sys.argv[1:])


def quantify():
    return _invoke("quantify", sys.argv[1:])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # Parse just the first token so downstream --help/options reach their parser.
    parser = argparse.ArgumentParser(prog="scTIER", description="scTIER integrates scASprofiler and scGIFT")
    parser.add_argument("--version", action="version", version=f"scTIER {__version__}")
    parser.add_argument("module", choices=["scASprofiler", "scGIFT", "scasprofiler", "scgift"])
    args = parser.parse_args(argv[:1])
    if args.module.lower() == "scgift":
        from .scGIFT.cli import main as gift_main
        return gift_main(argv[1:])
    return scasprofiler(argv[1:])
