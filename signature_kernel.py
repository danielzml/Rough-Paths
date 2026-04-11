import numpy as np
from signature import signatures_via_cde_batch


def refine_path_batch(paths, sub_steps):
    """Batched linear interpolation (FULLY VECTORIZED, NO LOOP).
    
    Args:
        paths: (B, L, d) batch of paths
        sub_steps: int
    Returns:
        (B, (L-1)*sub_steps + 1, d)
    """
    B, L, d = paths.shape

    p0 = paths[:, :-1, None, :]
    p1 = paths[:, 1:, None, :]
    
    alpha = np.linspace(0, 1, sub_steps)[None, None, :, None]
    
    refined_segments = p0 * (1 - alpha) + p1 * alpha
    
    refined_segments = refined_segments.reshape(B, -1, d)
    
    refined_path = np.concatenate([refined_segments, paths[:, -1:, :]], axis=1)
    
    return refined_path


def build_diagonal_indices(N, M):
    diagonals = []
    for d in range(N + M - 1):
        i_min = max(0, d - M + 1)
        i_max = min(d + 1, N)
        i = np.arange(i_min, i_max)
        j = d - i
        diagonals.append((i, j))
    return diagonals



def solve_signature_pde_batch(K, C, diagonals):
    B, Np1, Mp1 = K.shape
    K_new = np.ones_like(K)

    for i, j in diagonals:
        # all indices are arrays of the same shape
        K_new[:, i + 1, j + 1] = (
            K_new[:, i + 1, j] +
            K_new[:, i, j + 1] -
            K_new[:, i, j] +
            C[:, i, j] * K[:, i, j]
        )

    return K_new



def signature_kernel_batch(x_batch, y_batch, sub_steps, n_iter):
    B = x_batch.shape[0]

    x_ref = refine_path_batch(x_batch, sub_steps)
    y_ref = refine_path_batch(y_batch, sub_steps)

    dx = np.diff(x_ref, axis=1)
    dy = np.diff(y_ref, axis=1)

    N, M = dx.shape[1], dy.shape[1]

    C = np.einsum("bik,bjk->bij", dx, dy)

    K = np.ones((B, N + 1, M + 1))

    diagonals = build_diagonal_indices(N, M)


    for _ in range(n_iter):
        K = solve_signature_pde_batch(K, C, diagonals)

    return K[:, -1, -1]


def truncated_signature_kernel_batch(x_batch, y_batch, depth):
    """
    Compute truncated signature kernels using batched CDE.
    """
    
    # Compute all signatures at once (batched)
    sig_x = signatures_via_cde_batch(x_batch, depth = depth)  # (B, dim_T)
    sig_y = signatures_via_cde_batch(y_batch, depth = depth)  # (B, dim_T)

    # Element-wise dot products: <Sig(x_i), Sig(y_i)> for each i
    kernels = np.einsum('ij,ij->i', sig_x, sig_y)
    
    return kernels


# =====================
# Example usage
# =====================
if __name__ == "__main__":
    B = 40       # batch size
    L = 30       # path length
    d = 4        # path dimension
    scale = 0.2
    depth = 4


    np.random.seed(43)
    
    # Generate B pairs of paths
    x_batch = np.cumsum(scale * np.random.randn(B, L-1, d), axis=1)
    x_batch = np.concatenate([np.zeros((B, 1, d)), x_batch], axis=1)
    
    y_batch = np.cumsum(scale * np.random.randn(B, L-1, d), axis=1)
    y_batch = np.concatenate([np.zeros((B, 1, d)), y_batch], axis=1)
    
    # Compute all kernel values 
    full_kernels = signature_kernel_batch(x_batch, y_batch, sub_steps=20, n_iter=depth)
    truncated_kernels = truncated_signature_kernel_batch(x_batch, y_batch, depth)

    print(full_kernels)
    print(truncated_kernels)
