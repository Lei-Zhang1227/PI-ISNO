# burger2D

This folder contains the reproducibility package for the 2D Burgers equation case used in the PI-ISNO experiments.

## Directory Layout

```text
burger2D/
├── data/      # input data files for local reproduction
├── code/      # training, testing, model, loss, and data-loading scripts
├── result/    # archived checkpoint, logs, metrics, and figures for the selected run
└── yaml/      # experiment configuration files
```

## Selected Experiment

The archived run is:

```text
result/51-5-res-B
```

This is the selected best run for the 2D Burgers equation case. The corresponding configuration is:

```text
yaml/information2.yaml
```

The archived result folder includes:

- `checkpoint-best.pth.tar`: best model checkpoint for reproduction/evaluation.
- `checkpoint-best.pkl`: serialized best-checkpoint metadata/history saved by the original run.
- `Experiment_record.txt`: training log and experiment record.
- `test_results.csv`: summary metrics for test resolutions.
- `seen_extend_*.csv` and `unseen_extend_*.csv`: extension-test error records.
- `loss_carve_for_checkpoint-best_loss_analysis.png`: loss/history visualization for the best checkpoint.
- `figures_test_32/`, `figures_test_64/`, `figures_test_128/`: best/mid/worst qualitative prediction figures.

## Data

The packaged configuration uses paths relative to this case folder. For local reproduction, place the dataset in:

```text
data/burgers2d_spectral.h5
```

`yaml/information2.yaml` is already set to:

```yaml
data:
  datapath: 'data/burgers2d_spectral.h5'

prepare:
  project: 'result/51-5-res-B'
```

The Python entry points resolve these paths against the `burger2D/` case root, so the commands work when run from `code/`.

No real paper-scale `.h5` data file is currently included in this local package; only the tiny dummy HDF5 file for smoke testing is included.


## Path Convention

All active configuration paths are case-root relative:

- `data.datapath`: input HDF5 file under `data/`.
- `data.extend_datapath`: optional separate HDF5 file for `test_extend.py`; if omitted, `test_extend.py` uses `data.datapath`.
- `prepare.project`: output/result directory under `result/`.

The scripts define `CASE_ROOT = Path(__file__).resolve().parents[1]` and resolve relative paths from that root. This avoids machine-specific paths such as training-server `/data/...` or `/code/...` locations.

`result/51-5-res-B/Experiment_record.txt` is preserved as the original training log, so it still contains the original server paths for audit history. Those paths are historical records, not active local configuration.

## Reproduction Commands

Run commands from the `code/` directory unless paths are adjusted.

Train with the selected configuration:

```bash
cd code
python main.py --config_path ../yaml/information2.yaml --mode train
```

Evaluate using the archived best checkpoint:

```bash
cd code
python main.py --config_path ../yaml/information2.yaml --mode test --pretrain result/51-5-res-B/checkpoint-best.pth.tar
```

Extended testing, if needed:

```bash
cd code
python test_extend.py --config_path ../yaml/information2.yaml --mode test --pretrain result/51-5-res-B/checkpoint-best.pth.tar
```


## Local Smoke Test

A tiny synthetic HDF5 dataset and smoke-test configuration are included only to verify that the local code path runs:

```text
data/burgers2d_spectral_dummy.h5
yaml/smoke_information.yaml
code/make_dummy_data.py
code/smoke_test.py
```

Generate the dummy data and run the smoke test from `code/`:

```bash
python make_dummy_data.py
python smoke_test.py
```

The smoke test checks:

- HDF5 sample/group layout expected by `FNODatasetMult`.
- Train/test split behavior: 1 dummy training sample and 100 dummy test samples.
- Tensor shapes: `xx=(8,8,3,1)`, `yy=(8,8,11,1)`, `grid=(8,8,2)`.
- One `SOL2D` forward pass with a reduced model.

Current local result with `D:\anaconda\envs\pi-sol\python.exe`:

```text
SMOKE TEST PASSED
```

This dummy test is not a physics reproduction of the paper result; it only verifies that the packaged code can be imported, read HDF5 data, build the model, and complete a forward pass locally.

## Main Recorded Metrics

Metrics from `result/51-5-res-B/test_results.csv`:

| Dataset | Mean Relative L2 Error | Mean PDE Error |
| --- | ---: | ---: |
| test_128 | 0.0097311372 | 0.0087604377 |
| test_64 | 0.0077673396 | 0.0046593233 |
| test_32 | 0.0184797355 | 0.0068515285 |

## Notes

- The package intentionally keeps only the selected checkpoint and supporting history/results for the paper experiment.
- Intermediate epoch checkpoints, cache files, notebook checkpoints, and large visualization pickle files are excluded.
- Before public release, verify that the local data path and run commands work on a clean environment.


