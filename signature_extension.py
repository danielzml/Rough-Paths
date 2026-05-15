import numpy as np
from time import perf_counter
from tqdm import tqdm

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
    out[0] = 1
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


def chen_multiply_right_truncated(sig_a: np.ndarray, sig_b: np.ndarray, dim: int, depth: int, right_depth: int) -> np.ndarray:
    levels = []
    for n in range(depth + 1):
        level_n = np.zeros(dim**n)
        k_min = max(0, n - right_depth)
        for k in range(k_min, n + 1):
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
        refined[j] = (1 - alpha) * path[i] + alpha * path[i + 1]

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


def extended_signature_via_segments(path: np.ndarray, depth_piece: int, depth_full: int, num_segments: int) -> np.ndarray:
    dim = path.shape[1]
    refined_path = interpolate_piecewise_linear(path, num_segments=num_segments)
    segments_batch = np.stack([refined_path[i:i + 2] for i in range(num_segments)])
    all_sig_pieces = signatures_via_cde_batch(segments_batch, depth=depth_piece)

    product_extended = tensor_identity(dim, depth_full)
    for sig_piece in all_sig_pieces:
        product_extended = chen_multiply_right_truncated(
            product_extended,
            sig_piece,
            dim,
            depth_full,
            depth_piece,
        )

    return product_extended


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    dim = 4
    depth_full = 8 
    depth_piece = 3
    num_coarse_points = 16
    num_segments = 800
    num_trials = 20

    coarse_paths = np.cumsum(
        rng.normal(scale=0.35, size=(num_trials, num_coarse_points, dim)),
        axis=1,
    )

    direct_times = []
    extended_times = []
    first_A = None
    first_product = None

    for coarse_path in tqdm(coarse_paths, desc="Benchmarking", unit="path"):
        start = perf_counter()
        A = signatures_via_cde_batch(coarse_path[None, :, :], depth=depth_full)[0]
        direct_times.append(perf_counter() - start)

        start = perf_counter()
        product_extended = extended_signature_via_segments(
            coarse_path,
            depth_piece=depth_piece,
            depth_full=depth_full,
            num_segments=num_segments,
        )
        extended_times.append(perf_counter() - start)

        if first_A is None:
            first_A = A
            first_product = product_extended

    direct_times = np.array(direct_times)
    extended_times = np.array(extended_times)

    print("Latency summary (seconds)")
    print(f"  direct depth-{depth_full}: mean = {direct_times.mean():.6f}, std = {direct_times.std():.6f}")
    print(f"  extended depth-{depth_full} from depth-{depth_piece}: mean = {extended_times.mean():.6f}, std = {extended_times.std():.6f}")
    print(f"  speedup (direct / extended) = {direct_times.mean() / extended_times.mean():.6f}")
    print()

    print("First-path direct signature (lhs)")
    print(first_A)
    print()

    print("First-path extended Chen-product signature (rhs)")
    print(first_product)
    print()

    print_levelwise_error("First-path error: direct vs zero-extended Chen product", first_A, first_product, dim, depth_full)