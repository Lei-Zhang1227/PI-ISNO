import h5py

with h5py.File('F:/data/OLdata/heat_robin1k.mat', 'r') as f:
    print("Keys:", list(f.keys()))

    for key in f.keys():
        if isinstance(f[key], h5py.Dataset):
            print(f"{key}: shape={f[key].shape}, dtype={f[key].dtype}")