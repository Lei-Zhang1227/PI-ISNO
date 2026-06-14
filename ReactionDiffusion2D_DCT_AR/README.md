# ReactionDiffusion2D_DCT_AR

This case archives the 2D reaction-diffusion experiment corresponding to the `ex3` source folder.

## Layout

```text
ReactionDiffusion2D_DCT_AR/
|-- data/
|-- code/
|-- result/exp/
`-- yaml/
```

## Active Paths

- Configuration: `yaml/information.yaml`
- Training/test dataset: `data/2D_diff-react_NA_NA.h5`
- Extended evolution dataset: `data/2D_diff-react_full_0_15.h5`
- Long-time dataset placeholder: `data/2D_diff-react_t5_t15_101.h5`
- Selected result folder: `result/exp`
- Best checkpoint: `result/exp/checkpoint-best.pth.tar`

The original scripts depended on a shared server package named `NOs_dict`. A local compatibility wrapper is included under `code/NOs_dict/` so the archived scripts import from this case folder.

## Reproduction

```bash
cd code
python main_res.py --config_path ../yaml/information.yaml --mode train
python main_res.py --config_path ../yaml/information.yaml --mode test --pretrain result/exp/checkpoint-best.pth.tar
python test_extend.py --config_path ../yaml/information.yaml --pretrain result/exp/checkpoint-best.pth.tar
```

## Archived Metrics

| Dataset | Mean Relative L2 Error | Mean PDE Error |
| --- | ---: | ---: |
| test_128 | 0.0140945076 | 0.0005195843 |
| test_64 | 0.0767936077 | 0.0946670300 |
| test_32 | 0.2213337974 | 0.4159991810 |
| test_16 | 0.7979298747 | 1.5880171895 |
