from pathlib import Path
from functools import partial as PARTIAL
import yaml
import torch
import numpy as np

from dataloader import FNODatasetMult
from model import SOL2D, Transform, Wrapper, dctII, idctII

root = Path(__file__).resolve().parents[1]
config_path = root / "yaml" / "smoke_information.yaml"
with open(config_path, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

data_path = (root / "data" / "burgers2d_spectral_dummy.h5").resolve()
train_data = FNODatasetMult(file_path=str(data_path), initial_step=3, sub_x=1, sub_t=1, if_test=False)
test_data = FNODatasetMult(file_path=str(data_path), initial_step=3, sub_x=1, sub_t=1, if_test=True)
print(f"train samples: {len(train_data)}, test samples: {len(test_data)}")
assert len(train_data) == 1, len(train_data)
assert len(test_data) == 100, len(test_data)

xx, yy, grid = train_data[0]
print("single sample shapes:", tuple(xx.shape), tuple(yy.shape), tuple(grid.shape))
assert tuple(xx.shape) == (8, 8, 3, 1)
assert tuple(yy.shape) == (8, 8, 11, 1)
assert tuple(grid.shape) == (8, 8, 2)

loader = torch.utils.data.DataLoader(train_data, batch_size=1, shuffle=False)
xx, yy, grid = next(iter(loader))
print("batch shapes:", tuple(xx.shape), tuple(yy.shape), tuple(grid.shape))

T = Transform(PARTIAL(Wrapper, [dctII, dctII]), PARTIAL(Wrapper, [idctII, idctII]))
model = SOL2D(
    T,
    in_channels=config["train"]["initial_step"] + 3,
    modes=config["model"]["modes"],
    width=config["model"]["width"],
    bandwidth=config["model"]["bandwidth"],
    out_channels=config["model"]["output_channel"],
    dim=config["model"]["dim"],
    triL=config["model"]["triL"],
    double_weights=False,
    skip=True,
    flat=False,
)
model.eval()

current_time = torch.tensor([[[[0.3]]]], dtype=xx.dtype).expand(xx.size(0), xx.size(1), xx.size(2), 1)
inp = torch.cat([xx.squeeze(-1), grid, current_time], dim=-1)
print("model input shape:", tuple(inp.shape))
with torch.no_grad():
    out = model(inp)
print("model output shape:", tuple(out.shape))
assert tuple(out.shape) == (1, 8, 8, 1)
assert torch.isfinite(out).all()

print("SMOKE TEST PASSED")
