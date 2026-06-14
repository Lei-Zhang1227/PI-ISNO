# Heat1D_Robin_DCT_AR

This folder contains the reproducibility package for the 1D heat equation with Robin boundary condition used in the PI-ISNO experiments.

## Directory Layout

```text
Heat1D_Robin_DCT_AR/
├── data/      # input HDF5 data files for local reproduction
├── code/      # training, testing, model, loss, and data-loading scripts
├── result/    # archived checkpoint, logs, metrics, and figures for the selected run
└── yaml/      # experiment configuration files
```

## Selected Experiment

The archived run is:

```text
result/51-5-testforRES_repeat
```

The corresponding configuration is:

```text
yaml/information2.yaml
```

The archived result folder includes:

- `checkpoint-best.pth.tar`: best model checkpoint for reproduction/evaluation.
- `checkpoint-best.pkl`: serialized best-checkpoint metadata/history saved by the original run.
- `Experiment_record.txt`: original training log and experiment record.
- `test_results.csv`: summary metrics for test resolutions.
- `extend_results.csv`: extension-test error records.
- `loss_carve_for_checkpoint-best_loss_analysis.png`: loss/history visualization.
- `error_talk.png`, `sample_l2_summary.pkl`, `visualize_results.pkl`: selected analysis artifacts.
- `figures_test_65/`, `figures_test_129/`, `figures_test_257/`, `figures_test_513/`: qualitative prediction figures.

## Data

The active configuration uses paths relative to this case folder. For local reproduction, place the datasets in:

```text
data/heat1D_robin_highprec.h5
data/heat1D_robin_highprec_long.h5      # optional, used by test_extend.py
```

`yaml/information2.yaml` is set to:

```yaml
data:
  datapath: 'data/heat1D_robin_highprec.h5'
  extend_datapath: 'data/heat1D_robin_highprec_long.h5'

prepare:
  project: 'result/51-5-testforRES_repeat'
```

The Python entry points resolve relative paths against the `Heat1D_Robin_DCT_AR/` case root, so commands work when run from `code/`.

No paper-scale `.h5` data file is currently included in this local package.

## Path Convention

All active configuration paths are case-root relative:

- `data.datapath`: main input HDF5 file under `data/`.
- `data.extend_datapath`: optional long-time HDF5 file for `test_extend.py`.
- `prepare.project`: output/result directory under `result/`.

The scripts define `CASE_ROOT = Path(__file__).resolve().parents[1]` and resolve relative paths from that root. This avoids machine-specific training-server paths.

`result/51-5-testforRES_repeat/Experiment_record.txt` is preserved as the original training log, so it still contains original server paths for audit history. Those paths are historical records, not active local configuration.

## Reproduction Commands

Run commands from the `code/` directory unless paths are adjusted.

Train with the selected configuration:

```bash
cd code
python main_res.py --config_path ../yaml/information2.yaml --mode train
```

Evaluate using the archived best checkpoint:

```bash
cd code
python main_res.py --config_path ../yaml/information2.yaml --mode test --pretrain result/51-5-testforRES_repeat/checkpoint-best.pth.tar
```

Extended testing, if the long-time dataset is available:

```bash
cd code
python test_extend.py --config_path ../yaml/information2.yaml --mode test --pretrain result/51-5-testforRES_repeat/checkpoint-best.pth.tar
```

## Main Recorded Metrics

Metrics from `result/51-5-testforRES_repeat/test_results.csv`:

| Dataset | Mean Relative L2 Error | Mean PDE Error |
| --- | ---: | ---: |
| test_513 | 0.0001778868 | 0.4017343216 |
| test_257 | 0.0001778645 | 0.0451096808 |
| test_129 | 0.0001779619 | 0.0115763185 |
| test_65 | 0.0001826806 | 0.0081396785 |

## Notes

- The package intentionally keeps only the selected checkpoint and supporting history/results for the paper experiment.
- Intermediate epoch checkpoints, cache files, notebook checkpoints, and large temporary files are excluded.
- Before public release, verify the real HDF5 data files are present under `data/` and rerun the evaluation command in a clean environment.
