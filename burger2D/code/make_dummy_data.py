import h5py
import numpy as np
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root / "data" / "burgers2d_spectral_dummy.h5"
out.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(0)
t = np.linspace(0.0, 1.0, 11, dtype=np.float32)
x = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
y = np.linspace(-1.0, 1.0, 8, dtype=np.float32)
X, Y = np.meshgrid(x, y, indexing="ij")

with h5py.File(out, "w") as f:
    for i in range(101):
        g = f.create_group(f"sample_{i:04d}")
        phase = np.float32(i / 101.0)
        data = np.empty((len(t), len(x), len(y), 1), dtype=np.float32)
        for j, tj in enumerate(t):
            field = np.sin(np.pi * (X + phase)) * np.cos(np.pi * (Y - phase)) * np.exp(-tj)
            field += 0.01 * rng.standard_normal(field.shape).astype(np.float32)
            data[j, :, :, 0] = field
        g.create_dataset("data", data=data, compression="gzip")
        grid = g.create_group("grid")
        grid.create_dataset("x", data=x)
        grid.create_dataset("y", data=y)
print(out)
