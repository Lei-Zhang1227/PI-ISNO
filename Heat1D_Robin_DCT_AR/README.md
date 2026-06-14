# Heat1D_Robin_DCT_AR

This case archives the 1D heat equation experiment with Robin boundary conditions, DCT boundary handling, and autoregressive time marching.

## Layout

```text
Heat1D_Robin_DCT_AR/
|-- data/
|-- code/
|-- result/exp/
`-- yaml/
```

## Active Paths

- Configuration: `yaml/information2.yaml`
- Dataset: `data/heat_1100.h5`
- Selected result folder: `result/exp`
- Best checkpoint: `result/exp/checkpoint-best.pth.tar`

## Reproduction

```bash
cd code
python main_res.py --config_path ../yaml/information2.yaml --mode train
python main_res.py --config_path ../yaml/information2.yaml --mode test --pretrain result/exp/checkpoint-best.pth.tar
python test_extend.py --config_path ../yaml/information2.yaml --mode test_extend --pretrain result/exp/checkpoint-best.pth.tar
```

## Archived Metrics

Metrics are stored in `result/exp/test_results.csv`.
