# PI-ISNO Reproducibility Package

This repository is organized as a case-based reproducibility package for the PI-ISNO experiments. Each case keeps the minimum materials needed to support the reported experiment: code, selected configuration, archived best checkpoint, training/evaluation history, and result summaries.

The current public release contains six reproducibility case folders. Five cases include selected archived results/checkpoints, while the Burger1D result checkpoint and several full paper-scale datasets are being prepared and will be uploaded in subsequent updates.

## Cases

```text
pi-isno/
├── burger2D/
├── Burgers1D_Neumann_DCT_AR/
├── Heat1D_Robin_DCT_AR/
├── Heat2D_Neumann_HarmonicLifting/
├── AllenCahn2D_Neumann_HarmonicLifting/
└── ReactionDiffusion2D_DCT_AR/
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
| Burgers1D_Neumann_DCT_AR | `yaml/information.yaml` | `result/DCT-res-new-data-old_file` (checkpoint/results pending) |
| Heat1D_Robin_DCT_AR | `yaml/information2.yaml` | `result/51-5-testforRES_repeat` |
| Heat2D_Neumann_HarmonicLifting | `yaml/information.yaml` | `result/51-3-res-A` |
| AllenCahn2D_Neumann_HarmonicLifting | `yaml/information.yaml` | `result/51-5-res-A` |
| ReactionDiffusion2D_DCT_AR | `yaml/information.yaml` | `result/RA-RES-A` |

## Local Verification Status

- `burger2D` includes a tiny dummy HDF5 dataset and smoke test; it has been run locally with `D:\anaconda\envs\pi-sol\python.exe`.
- `Heat1D_Robin_DCT_AR` and `Heat2D_Neumann_HarmonicLifting` core scripts were checked with `python -m py_compile`; full evaluation requires the real HDF5 datasets under each case's `data/` folder.
- `AllenCahn2D_Neumann_HarmonicLifting` and `ReactionDiffusion2D_DCT_AR` include selected archived checkpoints, histories, metrics, and figures. Their main active yaml/script paths have been normalized to case-root relative paths.
- `Burgers1D_Neumann_DCT_AR` currently includes code, selected yaml, and the multi-resolution sampling table. The source folder did not contain the archived output/checkpoint files, so those will be added later.

## Release Notes

The package intentionally excludes intermediate epoch checkpoints, notebook checkpoints, caches, and very large temporary artifacts. The goal is to keep enough material to verify that each experiment is real and reproducible without uploading unnecessary training clutter.

Additional cases and dataset links/files will be added soon as the remaining experiments are organized for public release.
