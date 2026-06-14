# Heat2D_Neumann_HarmonicLifting

This case archives the 2D heat equation experiment with Neumann boundary conditions and harmonic lifting.

## Layout

```text
Heat2D_Neumann_HarmonicLifting/
|-- data/
|-- code/
|-- result/exp/
`-- yaml/
```

## Active Paths

- Configuration: `yaml/information.yaml`
- Dataset: `data/heat2d_neumann_1100.h5`
- Selected result folder: `result/exp`
- Best checkpoint: `result/exp/checkpoint-best.pth.tar`

The archived result folder includes metric CSV files, selected figures, boundary-condition verification outputs, and `loss_carve_for_checkpoint-best_loss_analysis.png`.

## Reproduction

```bash
cd code
python main.py --config_path ../yaml/information.yaml --mode train
python main.py --config_path ../yaml/information.yaml --mode test --pretrain result/exp/checkpoint-best.pth.tar
python test_extend.py --config_path ../yaml/information.yaml --mode test_extend --pretrain result/exp/checkpoint-best.pth.tar
```

## Archived Metrics

Metrics are stored in `result/exp/test_results.csv`.

## Figure Naming Note

Archived qualitative figure file names use the `heat2d_*` prefix. Some image canvases still contain the original plotting label "Burgers" because the figures are preserved from the training archive.
