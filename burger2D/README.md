# burger2D

This case archives the 2D Burgers equation experiment used in the PI-ISNO study.

## Layout

```text
burger2D/
|-- data/
|-- code/
|-- result/exp/
`-- yaml/
```

## Active Paths

- Configuration: `yaml/information2.yaml`
- Dataset: `data/burgers2d_spectral.h5`
- Selected result folder: `result/exp`
- Best checkpoint: `result/exp/checkpoint-best.pth.tar`

A tiny dummy dataset is also included for smoke testing: `data/burgers2d_spectral_dummy.h5`.

## Reproduction

```bash
cd code
python main.py --config_path ../yaml/information2.yaml --mode train
python main.py --config_path ../yaml/information2.yaml --mode test --pretrain result/exp/checkpoint-best.pth.tar
python test_extend.py --config_path ../yaml/information2.yaml --mode test --pretrain result/exp/checkpoint-best.pth.tar
```

## Smoke Test

```bash
cd code
python make_dummy_data.py
python smoke_test.py
```

## Archived Metrics

Metrics are stored in `result/exp/test_results.csv`.
