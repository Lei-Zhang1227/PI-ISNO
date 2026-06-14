# PI-ISNO Reproducibility Package

This repository is organized as a case-based reproducibility package for the PI-ISNO experiments. Each case keeps the minimum materials needed to support the reported experiment: code, selected configuration, archived best checkpoint, training/evaluation history, and result summaries.

The current public release contains three reproducibility cases. Additional benchmark cases and the corresponding full datasets are being prepared and will be uploaded in subsequent updates.

## Cases

```text
pi-isno/
├── burger2D/
├── Heat1D_Robin_DCT_AR/
└── Heat2D_Neumann_HarmonicLifting/
```

Each case follows the same layout:

```text
case_name/
├── data/      # local HDF5 datasets, not all paper-scale data are included
├── code/      # scripts required for training/evaluation
├── result/    # selected checkpoint, logs, metrics, and figures
├── yaml/      # selected experiment configuration
└── README.md  # case-specific reproduction notes
```

## Path Convention

Active yaml and code paths are case-root relative. For example, `data.datapath: data/example.h5` resolves to `case_name/data/example.h5`, and `prepare.project: result/run_name` resolves to `case_name/result/run_name`.

Historical training logs in `result/` are preserved as original records. They may still contain training-server paths such as `/data/...` or `/code/...`; those are audit history, not active local configuration.

## Selected Runs

| Case | Configuration | Selected result |
| --- | --- | --- |
| burger2D | `yaml/information2.yaml` | `result/51-5-res-B` |
| Heat1D_Robin_DCT_AR | `yaml/information2.yaml` | `result/51-5-testforRES_repeat` |
| Heat2D_Neumann_HarmonicLifting | `yaml/information.yaml` | `result/51-3-res-A` |

## Local Verification Status

- `burger2D` includes a tiny dummy HDF5 dataset and smoke test; it has been run locally with `D:\anaconda\envs\pi-sol\python.exe`.
- `Heat1D_Robin_DCT_AR` and `Heat2D_Neumann_HarmonicLifting` core scripts were checked with `python -m py_compile`; full evaluation requires the real HDF5 datasets under each case's `data/` folder.

## Release Notes

The package intentionally excludes intermediate epoch checkpoints, notebook checkpoints, caches, and very large temporary artifacts. The goal is to keep enough material to verify that each experiment is real and reproducible without uploading unnecessary training clutter.

More cases and dataset links/files will be added soon as the remaining experiments are organized for public release.
