# Burgers1D_Neumann_DCT_AR

This case archives the 1D Burgers equation experiment with non-homogeneous Neumann boundary conditions. The selected training setup uses DCT-based boundary handling and autoregressive time marching.

## Layout

```text
Burgers1D_Neumann_DCT_AR/
├── data/      # place the HDF5 dataset here
├── code/      # training/evaluation scripts and resolution_dfy.pkl
├── result/    # selected run output; checkpoint/results are pending for this case
└── yaml/      # selected experiment configuration
```

## Paths

The active yaml uses case-root relative paths:

- Dataset: `data/burgers_neumann_513x101_1s_old_format.h5`
- Selected result folder: `result/DCT-res-new-data-old_file`
- Multi-resolution sampling table: `code/resolution_dfy.pkl`

Historical absolute server paths have been removed from the selected yaml and the main reproduction script (`code/main_res.py`).

## Reproduction

From this case folder:

```bash
cd code
python main_res.py --config_path ../yaml/information.yaml --mode train
python main_res.py --config_path ../yaml/information.yaml --mode test --pretrain result/DCT-res-new-data-old_file/checkpoint-best.pth.tar
```

## Current Archive Status

The provided source folder contains the Burger1D code, selected yaml, and `resolution_dfy.pkl`, but no archived `output/` folder, `checkpoint-best.pth.tar`, or `test_results.csv` was found under `F:\pi-isno\WORK2\code`. The result directory is therefore prepared as the expected target location, and the checkpoint/history files will be added when the Burger1D archive is available.
