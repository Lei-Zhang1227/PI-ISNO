import numpy as np
from scipy.fft import dctn, idctn
from scipy.integrate import solve_ivp
import h5py
import time
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Extend Allen-Cahn dataset from 101 to 501 timesteps'
    )

    # Input/Output
    parser.add_argument('--input', type=str, 
                        default='/data/zhanglei/BurgersEquationII/ac2d_randbc_1100.h5',
                        help='Input HDF5 file (101 timesteps)')
    parser.add_argument('--output', type=str,
                        default='/data/zhanglei/BurgersEquationII/ac2d_extend_200.h5',
                        help='Output HDF5 file (501 timesteps)')
    parser.add_argument('--start_idx', type=int, default=0,
                        help='Starting sample index in input file (default: 0)')
    parser.add_argument('--n_samples', type=int, default=200,
                        help='Number of samples to extend (default: 200)')

    # PDE parameters (should match original dataset)
    parser.add_argument('--epsilon', type=float, default=0.05,
                        help='Interface thickness parameter (default: 0.05)')

    # Extended time parameters
    parser.add_argument('--T', type=float, default=5.0,
                        help='Final time for extended dataset (default: 5.0)')
    parser.add_argument('--Nt_save', type=int, default=501,
                        help='Number of time snapshots to save (default: 501)')

    # Solver selection
    parser.add_argument('--solver', type=str, default='imex',
                        choices=['rk45', 'rk23', 'bdf', 'lsoda', 'imex'],
                        help='ODE solver method (default: imex)')
    # For scipy solvers
    parser.add_argument('--rtol', type=float, default=1e-6,
                        help='Relative tolerance for scipy solvers (default: 1e-6)')
    parser.add_argument('--atol', type=float, default=1e-8,
                        help='Absolute tolerance for scipy solvers (default: 1e-8)')
    parser.add_argument('--max_step', type=float, default=0.05,
                        help='Max step size for scipy solvers (default: 0.05)')
    # For IMEX solver
    parser.add_argument('--dt', type=float, default=0.005,
                        help='Time step for IMEX solver (default: 0.005)')

    return parser.parse_args()


# ================================================================
# Lifting construction
# ================================================================

def build_lifting(X, Y, a, b, c, d):
    """Parametric harmonic lifting: Δu_b = 0."""
    alpha = (b - a) / 4.0
    beta = (d - c) / 4.0
    gamma = (a + b) / 2.0
    delta = (c + d) / 2.0
    return alpha * X**2 + beta * Y**2 + gamma * X + delta * Y


# ================================================================
# DCT-I spectral tools
# ================================================================

def build_eigenvalues(Nx, Ny, Lx, Ly):
    kx = np.arange(Nx, dtype=np.float64)
    ky = np.arange(Ny, dtype=np.float64)
    Kx, Ky = np.meshgrid(kx, ky, indexing='ij')
    return -((Kx * np.pi / Lx)**2 + (Ky * np.pi / Ly)**2)


def spectral_laplacian(u_h, Lambda):
    c = dctn(u_h, type=1)
    return idctn(c * Lambda, type=1)


# ================================================================
# Solvers
# ================================================================

def solve_scipy(u_h0, Lambda, u_b, epsilon, T, t_eval, Nx, Ny,
                method='RK45', rtol=1e-6, atol=1e-8, max_step=0.05):
    """Solve using scipy.integrate.solve_ivp.

    Returns U: (Nt, Nx, Ny) in u space (u = u_h + u_b).
    """
    def rhs(t, u_h_flat):
        u_h = u_h_flat.reshape(Nx, Ny)
        lap_u_h = spectral_laplacian(u_h, Lambda)
        u = u_h + u_b
        reaction = u - u**3
        return (epsilon * lap_u_h + reaction).ravel()

    sol = solve_ivp(
        rhs, [0, T], u_h0.ravel(),
        method=method, t_eval=t_eval,
        rtol=rtol, atol=atol, max_step=max_step,
    )

    if not sol.success:
        raise RuntimeError(f"Solver failed: {sol.message}")

    U_h = sol.y.reshape(Nx, Ny, len(t_eval))          # (Nx, Ny, Nt)
    U = (U_h + u_b[:, :, np.newaxis]).transpose(2, 0, 1)  # (Nt, Nx, Ny)
    return U.astype(np.float32)


def solve_imex(u_h0, Lambda, u_b, epsilon, T, t_eval, Nx, Ny, dt=0.005):
    """Solve using IMEX: Crank-Nicolson diffusion + explicit reaction.

    Crank-Nicolson step (in spectral space):
      (I - dt*ε/2 * Λ) ĉ^{n+1} = (I + dt*ε/2 * Λ) ĉ^n + dt * F̂(u^n)
    where F(u) = u - u³ is the reaction term.

    Returns U: (Nt, Nx, Ny) in u space (u = u_h + u_b).
    """
    Nt_internal = int(T / dt) + 1
    Nt_save = len(t_eval)

    # Precompute CN coefficients in spectral space
    half_dt_eps_Lambda = 0.5 * dt * epsilon * Lambda
    cn_numer = 1.0 + half_dt_eps_Lambda   # explicit side
    cn_denom = 1.0 - half_dt_eps_Lambda   # implicit side

    # Map t_eval to internal step indices
    save_indices = np.round(t_eval / dt).astype(int)
    save_indices = np.clip(save_indices, 0, Nt_internal - 1)

    U = np.zeros((Nt_save, Nx, Ny), dtype=np.float32)
    u_h = u_h0.copy()
    U[0] = (u_h + u_b).astype(np.float32)
    save_ptr = 1

    for n in range(1, Nt_internal):
        # Reaction term
        u = u_h + u_b
        f_n = u - u**3

        # CN step in spectral space
        c_h = dctn(u_h, type=1)
        rhs_spectral = cn_numer * c_h + dt * dctn(f_n, type=1)
        c_h_new = rhs_spectral / cn_denom
        u_h = idctn(c_h_new, type=1)

        # Save snapshot
        if save_ptr < Nt_save and n == save_indices[save_ptr]:
            U[save_ptr] = (u_h + u_b).astype(np.float32)
            save_ptr += 1

    return U


def solve_single(u_h0, Lambda, u_b, epsilon, T, t_eval, Nx, Ny, args):
    """Dispatch to the chosen solver.

    Returns U: (Nt, Nx, Ny) in u space.
    """
    if args.solver == 'imex':
        return solve_imex(u_h0, Lambda, u_b, epsilon, T, t_eval, Nx, Ny, dt=args.dt)
    else:
        method_map = {'rk45': 'RK45', 'rk23': 'RK23', 'bdf': 'BDF', 'lsoda': 'LSODA'}
        method = method_map[args.solver]
        return solve_scipy(u_h0, Lambda, u_b, epsilon, T, t_eval, Nx, Ny,
                           method=method, rtol=args.rtol, atol=args.atol,
                           max_step=args.max_step)


# ================================================================
# Main
# ================================================================

def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"Allen-Cahn Dataset Extension: 101 → 501 timesteps")
    print(f"{'='*60}")
    print(f"Input:   {args.input}")
    print(f"Output:  {args.output}")
    print(f"Samples: {args.n_samples} (indices {args.start_idx}-{args.start_idx + args.n_samples - 1})")
    print(f"Time:    [0, {args.T}], {args.Nt_save} snapshots")
    print(f"Solver:  {args.solver}", end='')
    if args.solver == 'imex':
        print(f" (dt={args.dt})")
    else:
        print(f" (rtol={args.rtol}, atol={args.atol}, max_step={args.max_step})")
    print(f"{'='*60}\n")

    # Open input file and read metadata
    with h5py.File(args.input, 'r') as f_in:
        # Read from first sample to get grid info
        first_key = f"{0:04d}"
        if first_key not in f_in:
            raise ValueError(f"Sample {first_key} not found in input file")
        
        x = f_in[first_key]['grid']['x'][:]
        y = f_in[first_key]['grid']['y'][:]
        Nx, Ny = len(x), len(y)
        Lx = float(x[-1] - x[0])
        Ly = float(y[-1] - y[0])
        
        print(f"Grid info from input:")
        print(f"  Nx={Nx}, Ny={Ny}")
        print(f"  Lx={Lx:.4f}, Ly={Ly:.4f}")
        print(f"  x range: [{x[0]:.4f}, {x[-1]:.4f}]")
        print(f"  y range: [{y[0]:.4f}, {y[-1]:.4f}]")
        print()

        # Check available samples
        available_samples = [int(k) for k in f_in.keys() if k.isdigit()]
        n_available = len(available_samples)
        print(f"Available samples in input: {n_available}")
        
        if args.n_samples > n_available:
            print(f"[WARNING] Requested {args.n_samples} samples but only {n_available} available")
            args.n_samples = n_available

    # Setup for extended simulation
    X, Y = np.meshgrid(x, y, indexing='ij')
    t_eval = np.linspace(0, args.T, args.Nt_save)
    Lambda = build_eigenvalues(Nx, Ny, Lx, Ly)

    # Process samples
    total_start = time.time()
    n_failed = 0

    with h5py.File(args.input, 'r') as f_in, h5py.File(args.output, 'w') as f_out:
        for out_idx in range(args.n_samples):
            sample_start = time.time()
            
            # 从input读取的索引
            in_idx = args.start_idx + out_idx
            in_key = f"{in_idx:04d}"
            
            if in_key not in f_in:
                print(f"\n[WARNING] Sample {in_idx} not found in input, skipping")
                n_failed += 1
                continue

            # Read initial condition (u space)
            u_0 = f_in[in_key]['data'][0, :, :, 0]  # (Nx, Ny)
            
            # Read BC parameters
            bc = f_in[in_key]['bc']
            a = float(bc['a'][()])
            b = float(bc['b'][()])
            c = float(bc['c'][()])
            d = float(bc['d'][()])
            
            # Construct lifting
            u_b = build_lifting(X, Y, a, b, c, d)
            
            # Extract u_h0 = u_0 - u_b
            u_h0 = u_0 - u_b

            # Solve with extended time
            try:
                U = solve_single(u_h0, Lambda, u_b, args.epsilon, args.T,
                                 t_eval, Nx, Ny, args)
            except RuntimeError as e:
                n_failed += 1
                print(f"\n[WARNING] Sample {in_idx} (input) → {out_idx} (output) failed: {e}")
                continue

            # Write HDF5 (输出索引从0开始)
            U_out = U[:, :, :, np.newaxis].astype(np.float32)  # (Nt, Nx, Ny, 1)

            out_key = f"{out_idx:04d}"
            grp = f_out.create_group(out_key)
            grp.create_dataset('data', data=U_out, dtype='float32')

            bc_grp = grp.create_group('bc')
            bc_grp.create_dataset('a', data=np.float32(a))
            bc_grp.create_dataset('b', data=np.float32(b))
            bc_grp.create_dataset('c', data=np.float32(c))
            bc_grp.create_dataset('d', data=np.float32(d))

            grid_grp = grp.create_group('grid')
            grid_grp.create_dataset('t', data=t_eval.astype(np.float32), dtype='float32')
            grid_grp.create_dataset('x', data=x.astype(np.float32), dtype='float32')
            grid_grp.create_dataset('y', data=y.astype(np.float32), dtype='float32')

            sample_time = time.time() - sample_start
            elapsed = time.time() - total_start
            eta = elapsed / (out_idx + 1) * (args.n_samples - out_idx - 1)

            print(f"\rSample {out_idx + 1:4d}/{args.n_samples} (input idx {in_idx}) | "
                  f"BC=({a:+.2f},{b:+.2f},{c:+.2f},{d:+.2f}) | "
                  f"{sample_time:.1f}s | "
                  f"已用: {elapsed / 60:.1f}min | "
                  f"剩余: {eta / 60:.1f}min | "
                  f"|u|_max: {np.abs(U).max():.3f}", end='')

    total_time = time.time() - total_start
    print(f"\n\n完成! 总用时: {total_time / 60:.1f} 分钟")
    if n_failed > 0:
        print(f"[WARNING] {n_failed} samples failed and were skipped")
    print(f"输出文件: {args.output}")
    print(f"文件大小: {os.path.getsize(args.output) / 1e9:.2f} GB")

    # Simple verification
    print(f"\n{'='*60}")
    print("Verification")
    print(f"{'='*60}")
    with h5py.File(args.output, 'r') as f:
        sample_key = f"{0:04d}"
        if sample_key in f:
            data_shape = f[sample_key]['data'].shape
            t_shape = f[sample_key]['grid']['t'].shape
            print(f"Output sample 0 data shape: {data_shape}")
            print(f"Output sample 0 time shape: {t_shape}")
            print(f"Expected: ({args.Nt_save}, {Nx}, {Ny}, 1)")
            
            # Check consistency with input
            in_sample_key = f"{args.start_idx:04d}"
            with h5py.File(args.input, 'r') as f_in:
                if in_sample_key in f_in:
                    u_in_0 = f_in[in_sample_key]['data'][0, :, :, 0]
                    u_out_0 = f[sample_key]['data'][0, :, :, 0]
                    diff = np.abs(u_in_0 - u_out_0).max()
                    print(f"\nInitial condition consistency check:")
                    print(f"  Comparing output[0] with input[{args.start_idx}]")
                    print(f"  Max difference: {diff:.6e}")
                    if diff < 1e-5:
                        print(f"  ✓ IC matches input dataset")
                    else:
                        print(f"  ✗ WARNING: IC differs from input dataset!")


if __name__ == '__main__':
    main()