import numpy as np
from signature import signature_via_cde


# ==========
# Parameters
# ==========

dim = 5
length = 50
scale = 0.2
depths = [1, 2, 3, 4, 5, 6]


def microscopic_path(length, dim, scale):
    """
    Generate a path with small increments.
    """
    increments = scale * np.random.randn(length - 1, dim)
    path = np.vstack([np.zeros(dim), np.cumsum(increments, axis=0)])
    return path


def sig_length(d, depth):
    return sum(d**k for k in range(depth + 1))


# ==========================
# Generate microscopic paths
# ==========================

x = microscopic_path(length, dim, scale)
y = microscopic_path(length, dim, scale)


print("Path dimension      :", dim)
print("Path length         :", length)
print("Path scale          :", scale)
print("\nSignature kernel values:")
print("=" * 40)

max_depth = max(depths)
sig_x_full = signature_via_cde(x, d=dim, depth=max_depth)
sig_y_full = signature_via_cde(y, d=dim, depth=max_depth)


# ================
# Loop over depths
# ================

for depth in depths:
    L = sig_length(dim, depth)

    sig_x = sig_x_full[:L]
    sig_y = sig_y_full[:L]

    dot_product = np.dot(sig_x, sig_y)

    print(f"Depth {depth:2d} | ⟨Sig(X), Sig(Y)⟩ = {dot_product:10.6f}")
   

def refine_path(path, sub_steps):
    new_points = []
    for i in range(len(path) - 1):
        p0, p1 = path[i], path[i+1]
        for k in range(sub_steps):
            alpha = k / sub_steps
            new_points.append(p0 * (1 - alpha) + p1 * alpha)
    new_points.append(path[-1])
    return np.array(new_points)


def solve_signature_pde(K, C, N, M):
    """
    Wavefront solver
    """
    K_new = np.ones((N + 1, M + 1))
    
    # The number of diagonals is (N + M - 1)
    for d in range(N + M - 1):
        # Identify the range of 'i' indices for the current diagonal i + j = d
        i_min = max(0, d - M + 1)
        i_max = min(d + 1, N)
        
        i = np.arange(i_min, i_max)
        j = d - i
        
        # Vectorized update for the entire diagonal at once
        K_new[i+1, j+1] = K_new[i+1, j] + K_new[i, j+1] - K_new[i, j] + C[i, j] * K[i, j]
        
    return K_new


def signature_kernel_infinite(x, y, sub_steps=50, n_iter=10):
    """
    Handles setup, refinement, and monitoring.
    """
    # Path Refinement
    x_refined = refine_path(x, sub_steps)
    y_refined = refine_path(y, sub_steps)

    # Compute path increments
    dx = np.diff(x_refined, axis=0)
    dy = np.diff(y_refined, axis=0)

    N, M = len(dx), len(dy)

    # Precompute Gram matrix of increments
    C = dx @ dy.T

    # Initialise Kernel matrix (Boundary conditions = 1)
    K = np.ones((N + 1, M + 1))

    # Solve PDE
    for iteration in range(1, n_iter + 1):
        K = solve_signature_pde(K, C, N, M)
        print(f"Iteration {iteration:2d} | ⟨Sig(X), Sig(Y)⟩ = {K[-1, -1]:.6f}")
    
    return K[-1, -1]


# ===============================
# Infinite depth signature kernel
# ===============================

full_signature_kernel = signature_kernel_infinite(x, y)
print(f"Depth  ∞ | ⟨Sig(X), Sig(Y)⟩ = {full_signature_kernel:10.6f}")