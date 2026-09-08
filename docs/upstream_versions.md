# Upstream Implementations

This benchmark adapts existing neural-operator implementations to the AhmedML
surface CFD dataset.

## Transolver-3

- Repository: https://github.com/thuml/Transolver-3
- Commit: `ef4fee9fa08dbfc5af13f9d9b42202dfb34dba37`
- Upstream branch: `main`

The AhmedML data pipeline, distributed training utilities, and Transolver-3
baseline were developed from this version.

## LRSA-Operator

- Repository: https://github.com/Adversarr/LRSA-Operator
- Commit: `47b03f8c8c8da30bbcc0737b008dc4548f9cb98e`

The upstream LRSA-Operator repository was used without local modifications.
`models/LRSA_chunk_opt_matrix_mul.py` provides the AhmedML-specific wrapper.

## LinearNO

The AhmedML LinearNO implementation is contained in
`models/LinearNO_chunk_opt_matrix_mul.py` and was adapted from the official
LinearNO implementation. Exact upstream repository/version information should
be recorded here before release.
