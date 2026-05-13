import numpy as np

from signature import signatures_via_cde_batch


def level_offsets(dim: int, depth: int):
    offsets = [0]
    for k in range(depth):
        offsets.append(offsets[-1] + dim**k)
    return offsets


def get_level(sig: np.ndarray, dim: int, level: int) -> np.ndarray:
    offsets = level_offsets(dim, level + 1)
    start = offsets[level]
    end = start + dim**level
    return sig[start:end]


def tensor_identity(dim: int, depth: int) -> np.ndarray:
    total_dim = sum(dim**k for k in range(depth + 1))
    out = np.zeros(total_dim)
    out[0] = 1.0
    return out


def chen_multiply(sig_a: np.ndarray, sig_b: np.ndarray, dim: int, depth: int) -> np.ndarray:
    levels = []
    for n in range(depth + 1):
        level_n = np.zeros(dim**n)
        for k in range(n + 1):
            a_k = get_level(sig_a, dim, k)
            b_nk = get_level(sig_b, dim, n - k)
            level_n += np.kron(a_k, b_nk)
        levels.append(level_n)
    return np.concatenate(levels)


def extend_depthp_to_depthq(sig_p: np.ndarray, dim: int, depth_p: int, depth_q: int) -> np.ndarray:
    if depth_q < depth_p:
        raise ValueError("depth_q must be at least depth_p")

    total_dim_p = sum(dim**k for k in range(depth_p + 1))
    total_dim_q = sum(dim**k for k in range(depth_q + 1))

    if sig_p.shape[0] != total_dim_p:
        raise ValueError(f"Expected signature of length {total_dim_p}, got {sig_p.shape[0]}")

    out = np.zeros(total_dim_q)
    out[:total_dim_p] = sig_p
    return out


def interpolate_piecewise_linear(path: np.ndarray, num_segments: int) -> np.ndarray:
    num_points, dim = path.shape
    grid = np.linspace(0.0, num_points - 1, num_segments + 1)

    refined = np.zeros((num_segments + 1, dim))
    for j, u in enumerate(grid):
        if np.isclose(u, num_points - 1):
            refined[j] = path[-1]
            continue

        i = int(np.floor(u))
        alpha = u - i
        refined[j] = (1.0 - alpha) * path[i] + alpha * path[i + 1]

    return refined


def print_levelwise_error(name: str, x: np.ndarray, y: np.ndarray, dim: int, depth: int):
    print(name)
    for level in range(depth + 1):
        x_level = get_level(x, dim, level)
        y_level = get_level(y, dim, level)
        err = np.linalg.norm(x_level - y_level)
        print(f"  level {level}: error = {err:.6e}")
    print(f"  total error: {np.linalg.norm(x - y):.6e}")
    print()


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    dim = 4
    depth_full = 7
    depth_piece = 3
    num_coarse_points = 10
    num_segments = 400

    # 1. Generate path
    coarse_path = np.cumsum(rng.normal(scale=0.35, size=(num_coarse_points, dim)), axis=0)
    
    # 2. Compute signature of path
    A = signatures_via_cde_batch(coarse_path[None, :, :], depth=depth_full)[0]

    # 3. Refine path
    refined_path = interpolate_piecewise_linear(coarse_path, num_segments=num_segments)
    
    # 4. Prepare all segments into a single batch: [num_segments, 2, dim]
    # Each slice [i:i+2] is a segment from point i to i+1
    segments_list = [refined_path[i : i + 2] for i in range(num_segments)]
    segments_batch = np.stack(segments_list) 

    # 5. Compute all signatures in a single call to the CDE solver
    print(f"Solving CDE for batch of {num_segments} segments...")
    all_sig_pieces = signatures_via_cde_batch(segments_batch, depth=depth_piece)
    
    # 6. Chain signatures together using Chen's relation
    product_extended = tensor_identity(dim, depth_full)
    
    for i in range(num_segments):
        sig_piece = all_sig_pieces[i] # Get the pre-computed signature for this segment
        sig_piece_extended = extend_depthp_to_depthq(sig_piece, dim, depth_piece, depth_full)
        product_extended = chen_multiply(product_extended, sig_piece_extended, dim, depth_full)

    # Print results 
    print_levelwise_error("Compare A to batched-extended-product", A, product_extended, dim, depth_full)
    