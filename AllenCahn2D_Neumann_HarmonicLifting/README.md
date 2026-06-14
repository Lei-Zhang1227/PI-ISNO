# AllenCahn2D_Neumann_HarmonicLifting

This case archives the 2D Allen-Cahn equation experiment with non-homogeneous Neumann boundary conditions. The selected run is `51-5-res-A`.

## Layout

```text
AllenCahn2D_Neumann_HarmonicLifting/
├── data/      # place HDF5 datasets here
├── code/      # training/evaluation scripts
├── result/    # selected checkpoint, logs, metrics, and figures
└── yaml/      # selected experiment configuration
```

## Paths

The active yaml and main scripts use case-root relative paths:

- Training/test dataset: `data/ac2d_randbc_1100.h5`
- Extended test dataset: `data/ac2d_extend_200.h5`
- Selected result folder: `result/51-5-res-A`
- Best checkpoint: `result/51-5-res-A/checkpoint-best.pth.tar`

Historical logs in `result/` are kept as original audit records and may still mention training-server paths.

The original `visualize_results.pkl` file is larger than GitHub's 100 MB single-file limit, so it is not included in this public repository. The checkpoint, metric CSV files, histories, and exported figures are retained.

## Reproduction

From this case folder:

```bash
cd code
python main.py --config_path ../yaml/information.yaml --mode train
python main.py --config_path ../yaml/information.yaml --mode test --pretrain result/51-5-res-A/checkpoint-best.pth.tar
python test_extend.py --config_path ../yaml/information.yaml --pretrain result/51-5-res-A/checkpoint-best.pth.tar
```

## Archived Metrics

| Dataset | Mean Relative L2 Error | Mean PDE Error |
| --- | ---: | ---: |
| test_129 | 0.0006033335 | 0.2419847618 |
| test_65 | 0.0006089428 | 0.2432040950 |
| test_33 | 0.0006767120 | 0.2456275862 |
