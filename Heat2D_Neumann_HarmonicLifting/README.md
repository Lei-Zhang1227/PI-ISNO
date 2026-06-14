# Heat2D_Neumann_HarmonicLifting

This folder contains the reproducibility package for the 2D heat equation with Neumann boundary condition and harmonic lifting used in the PI-ISNO experiments.

## Directory Layout

```text
Heat2D_Neumann_HarmonicLifting/
├── data/      # input HDF5 data files for local reproduction
├── code/      # training, testing, model, loss, and data-loading scripts
├── result/    # archived checkpoint, logs, metrics, and figures for the selected run
└── yaml/      # experiment configuration files
```

## Selected Experiment

The archived run is:

```text
result/51-3-res-A
```

The corresponding configuration is:

```text
yaml/information.yaml
```

The archived result folder includes:

- `checkpoint-best.pth.tar`: best model checkpoint for reproduction/evaluation.
- `checkpoint-best.pkl`: serialized best-checkpoint metadata/history saved by the original run.
- `Experiment_record.txt`: original training log and experiment record.
- `test_results.csv`: summary metrics for test resolutions.
- `seen_extend_*.csv` and `unseen_extend_*.csv`: extension-test error records.
- `timestep_errors*.pkl`: timestep-wise error summaries.
- `visualize_results.pkl`: selected visualization data.
- `loss_carve_for__code_heat2D_output_51_3_res_A_checkpoint_best_loss_analysis.png`: loss/history visualization.
- `figures_test_33/`, `figures_test_65/`, `figures_test_129/`: qualitative prediction figures.
- `fixed_bc_verification/` and `bc_generalization_test/`: selected boundary-condition verification outputs.

## Data

The active configuration uses paths relative to this case folder. For local reproduction, place the main dataset in:

```text
data/heat2d_neumann_1100.h5
```

Optional boundary-condition test scripts may also use:

```text
data/heat2d_bc_gen_test.h5
data/heat2d_param_bc.h5
data/heat2d_change_neumann_1100.h5
```

`yaml/information.yaml` is set to:

```yaml
data:
  datapath: 'data/heat2d_neumann_1100.h5'

prepare:
  project: 'result/51-3-res-A'
```

The Python entry points resolve relative paths against the `Heat2D_Neumann_HarmonicLifting/` case root, so commands work when run from `code/`.

No paper-scale `.h5` data file is currently included in this local package.

## Path Convention

All active configuration paths are case-root relative:

- `data.datapath`: main input HDF5 file under `data/`.
- `prepare.project`: output/result directory under `result/`.
- Auxiliary BC scripts also resolve their default HDF5 files from the case-root `data/` folder.

The scripts define `CASE_ROOT = Path(__file__).resolve().parents[1]` and resolve relative paths from that root. This avoids machine-specific training-server paths.

`result/51-3-res-A/Experiment_record.txt` and selected archived report files are preserved as original records, so they may still contain original server paths for audit history. Those paths are historical records, not active local configuration.

## Reproduction Commands

Run commands from the `code/` directory unless paths are adjusted.

Train with the selected configuration:

```bash
cd code
python main.py --config_path ../yaml/information.yaml --mode train
```

Evaluate using the archived best checkpoint:

```bash
cd code
python main.py --config_path ../yaml/information.yaml --mode test --pretrain result/51-3-res-A/checkpoint-best.pth.tar
```

Extended testing, if needed:

```bash
cd code
python test_extend.py --config_path ../yaml/information.yaml --mode test_extend --pretrain result/51-3-res-A/checkpoint-best.pth.tar
```

## Main Recorded Metrics

Metrics from `result/51-3-res-A/test_results.csv`:

| Dataset | Mean Relative L2 Error | Mean PDE Error |
| --- | ---: | ---: |
| test_129 | 0.0000338939 | 0.0001388778 |
| test_65 | 0.0000282343 | 0.0000931914 |
| test_33 | 0.0000777467 | 0.0001947735 |

## Notes

- The package intentionally keeps only the selected checkpoint and supporting history/results for the paper experiment.
- Intermediate epoch checkpoints, cache files, notebook checkpoints, and very large temporary visualization files are excluded.
- Before public release, verify the real HDF5 data files are present under `data/` and rerun the evaluation command in a clean environment.
