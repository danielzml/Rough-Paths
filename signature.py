import numpy as np
from scipy.integrate import solve_ivp


def signature_via_cde(path, d, depth):
    """
    Solve dS/dt = S ⊗ dX/dt in the truncated tensor algebra
    """
    dim_T = sum(d**k for k in range(depth+1))
   
    # Precompute offsets once
    offsets = [0] + [sum(d**i for i in range(k+1)) for k in range(depth)]
   
    def cde_rhs(t, S_flat):
        idx = min(int(t * (len(path)-1)), len(path)-2)
        X_dot = (path[idx+1] - path[idx]) * (len(path)-1)
       
        result = np.zeros_like(S_flat)
       
        # Only l=1 contributes since control has only level-1 non-zero
        for k in range(depth):
            start_in = offsets[k]
            size = d**k
            start_out = offsets[k+1]
           
            S_k = S_flat[start_in:start_in+size]
            # S^k ⊗ X_dot contributes to level k+1
            result[start_out:start_out+size*d] += np.outer(S_k, X_dot).ravel()
       
        return result
   
    S0 = np.zeros(dim_T)
    S0[0] = 1
   
    sol = solve_ivp(cde_rhs, [0, 1], S0, dense_output=True,
                    method='RK45', rtol=1e-10, atol=1e-12)
   
    return sol.y[:, -1]


def format_signature_by_levels(signature, d, depth):
    """
    Organise signature elements into a dictionary by tensor levels.
   
    Args:
        signature: flat signature vector
        d: dimension of the path
        depth: maximum level of the signature
   
    Returns:
        dict with keys 'level_0', 'level_1', ..., 'level_depth'
        Each level contains the corresponding signature terms
    """
    result = {}
    idx = 0
   
    for k in range(depth + 1):
        level_size = d ** k
        level_terms = signature[idx : idx + level_size]
       
        if k == 0:
            result['level_0'] = {
                'description': 'empty word',
                'terms': level_terms,
                'indices': [0]
            }
        else:
            # Generate word labels for this level
            # For level k, words are k-tuples of {1, 2, ..., d}
            words = []
            def generate_words(current, remaining):
                if remaining == 0:
                    words.append(''.join(map(str, current)))
                    return
                for i in range(1, d + 1):
                    generate_words(current + [i], remaining - 1)
           
            generate_words([], k)
           
            result[f'level_{k}'] = {
                'description': f'words of length {k}',
                'words': words,
                'terms': level_terms,
                'indices': list(range(idx, idx + level_size))
            }
       
        idx += level_size
   
    return result


def print_signature_dict(sig_dict, precision=5):
    """
    Pretty print the signature dictionary.
    """
    print("=" * 60)
    print("SIGNATURE")
    print("=" * 60)
   
    for level_key in sorted(sig_dict.keys()):
        level_data = sig_dict[level_key]
        k = int(level_key.split('_')[1])
       
        # Format: "level 0", "level 1", etc. (no uppercase, no underscore)
        display_key = f"level {k}"
       
        print(f"\n{'─' * 60}")
        print(f"  {display_key}: {level_data['description']}")
        print(f"{'─' * 60}")
       
        if k == 0:
            print(f"  S^∅ = {level_data['terms'][0]:.{precision}f}")
        else:
            words = level_data['words']
            terms = level_data['terms']
            for word, value in zip(words, terms):
                print(f"  S^{word} = {value:.{precision}f}")
   
    print(f"\n{'=' * 60}")


# Define path
path = np.array([
    [0.0, 0.0, 0.0],
    [0.3, 0.2, 0.1],
    [6.4, 0.5, 5.2],
    [0.2, 0.7, 0.3],
    [-0.2, 5.6, -0.4],
    [-4.5, 0.3, 0.5],
    [-0.6, -0.1, 0.6],
    [-1.4, -0.4, 0.4],
    [-9.1, -0.6, 5.3],
    [0.2, -1.5, 2.2],
    [0.4, -0.2, 0.1],
    [0.3, 0.3, 0.4],
    [5.4, 10, 2.2],
    [3.0, 2.0, 7.3],
    [2.3, 3.2, 7.1],
    [6.4, 0.5, 5.2],
    [0.2, 0.7, 6.3],
    [-0.2, 5.6, -0.4],
    [-4.5, 0.3, 0.5],
    [-8.3, -0.1, 9.4],
    [-1.4, -0.4, 0.4],
    [-5.1, -0.6, 4.3],
    [0.2, -1.5, 2.2],
    [9.4, -0.2, 0.1],
    [0.3, 0.3, 0.4],
    [5.4, 10, 2.2],
    [6.7, -15, 7.7],
    [0.8, -4.9, 7.6],
    [-7.1, 3.8, -9.4]
])

# Path dimension
d = path.shape[1]

# Compute signature
depth = 3
signature = signature_via_cde(path, d, depth)

# Organise and print by levels
sig_dict = format_signature_by_levels(signature, d, depth=depth)
print_signature_dict(sig_dict)
