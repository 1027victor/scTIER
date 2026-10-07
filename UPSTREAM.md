# Source provenance

scASprofiler is bundled from https://github.com/1027victor/scASprofiler
at commit eee26238ffb994b68a20a95088bb475b6b8be97f. Its Apache-2.0 LICENSE is retained. Its package location changes, and trailing whitespace is normalized. A small compatibility fix in
`scASp_perp/thresholding.py` returns the expected empty output schema when a
splicing direction has no groups, avoiding `pd.concat([])` crashes in plate preprocessing.

scGIFT originates from `iso_topic_fusion.py` supplied by the repository owner.
Original SHA256: 2a7ec0fd28610b781892c80ca6a92c2dbdd3004f2f53aade6820487f296cb68c.
Packaging changes: separate CLI and model, delay deterministic settings until execution,
validate arguments and input arrays, add explicit CPU/CUDA selection, and remove inactive
experimental comments and local sample paths. Active model, training objective, defaults,
and output filenames are retained.
