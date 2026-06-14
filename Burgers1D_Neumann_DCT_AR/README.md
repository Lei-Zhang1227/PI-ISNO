# Burgers1D_Neumann_DCT_AR

This case archives the 1D Burgers equation experiment with non-homogeneous Neumann boundary conditions. The selected setup uses DCT-based boundary handling and autoregressive time marching.

## Layout

```text
Burgers1D_Neumann_DCT_AR/
|-- data/
|-- code/
|-- result/exp/
`-- yaml/
```

## Active Paths

- Configuration: `yaml/information.yaml`
- Dataset: `data/burgers_neumann_513x101_1s_old_format.h5`
- Selected result folder: `result/exp`
- Multi-resolution sampling table: `code/resolution_dfy.pkl`

## Reproduction

```bash
cd code
python main_res.py --config_path ../yaml/information.yaml --mode train
python main_res.py --config_path ../yaml/information.yaml --mode test --pretrain result/exp/checkpoint-best.pth.tar
```

## Current Archive Status

The provided source folder contains the Burger1D code, selected yaml, and `resolution_dfy.pkl`, but no archived output folder, best checkpoint, or `test_results.csv` was found under `F:\pi-isno\WORK2\code`. The `result/exp` directory is prepared as the target location for the future Burger1D checkpoint and history files.
