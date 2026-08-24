

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt


# r = 5
# # data = np.load(f'./burgers/exact_intrusive_operator/exact_operator_reproj_{r}.npz')
# data = np.load(f'./burgers/exact_intrusive_operator/exact_operator_reproj_noise40_{r}.npz')

# A_r_true = data['A_r_true']
# H_r_true = data['H_r_true']
# Q_reproj = data['Q_reproj']
# t = data['t']




# def quadratic_snapshots(Q):
#     """Q: shape (r, K). Return shape (r*r, K)."""
#     return np.column_stack([
#         np.kron(Q[:, k], Q[:, k])
#         for k in range(Q.shape[1])
#     ])

# dt = t[1]-t[0]

# Q0 = Q_reproj[:, :-1]
# Q1 = Q_reproj[:, 1:]
# Q2 = quadratic_snapshots(Q0)

# Qdot = (Q1 - Q0) / dt

# Q1_true = (
#     Q0
#     + dt * (
#         A_r_true @ Q0
#         + H_r_true @ Q2
#     )
# )


# residual = Q1 - Q1_true

# relative_residual = (
#     np.linalg.norm(residual, "fro")
#     / np.linalg.norm(Q1, "fro")
# )

# print("true one-step relative residual:", relative_residual)















def compact_quadratic_snapshots(Q):
    """
    Construct unique quadratic monomials q_i*q_j with i <= j.

    Parameters
    ----------
    Q : ndarray, shape (r, m)
        Reduced state snapshots.

    Returns
    -------
    Q2 : ndarray, shape (r*(r+1)//2, m)
        Unique quadratic features.
    pairs : list of tuple
        Ordering of the quadratic monomials.
    """
    Q = np.asarray(Q, dtype=float)

    if Q.ndim != 2:
        raise ValueError("Q must have shape (r, m).")

    r, _ = Q.shape

    pairs = [
        (i, j)
        for i in range(r)
        for j in range(i, r)
    ]

    Q2 = np.vstack([
        Q[i, :] * Q[j, :]
        for i, j in pairs
    ])

    return Q2, pairs


def build_opinf_design_matrix(Q_reproj):
    """
    Construct the derivative-fitting OpInf design matrix.

    Parameters
    ----------
    Q_reproj : ndarray, shape (r, K)
        Re-projected reduced trajectory.

    Returns
    -------
    D : ndarray, shape (K-1, p)
        Least-squares design matrix.
    Phi : ndarray, shape (p, K-1)
        Column-oriented feature matrix.
    pairs : list of tuple
        Ordering of quadratic features.
    """
    Q0 = Q_reproj[:, :-1]

    Q2, pairs = compact_quadratic_snapshots(Q0)

    # Phi[:, k] = phi(q_k)
    Phi = np.vstack([
        Q0,
        Q2,
    ])

    # Each row of D corresponds to one snapshot.
    D = Phi.T

    return D, Phi, pairs

r = 5
noise = 20
# data = np.load(f"./burgers/exact_intrusive_operator/exact_operator_reproj_{r}.npz")
data = np.load(f"./burgers/exact_intrusive_operator/exact_operator_reproj_noise{noise}_{r}.npz")




Q_reproj = data["Q_reproj"]
t = data["t"]

D, Phi, pairs = build_opinf_design_matrix(Q_reproj)

print("Q_reproj shape:", Q_reproj.shape)
print("D shape:", D.shape)
print("Quadratic pairs:", pairs)




def numerical_rank_diagnostics(D, relative_tolerance=1e-12):
    """
    Compute numerical rank and condition diagnostics.

    Parameters
    ----------
    D : ndarray
        Design matrix.
    relative_tolerance : float
        Singular values satisfying
        sigma_i / sigma_max > relative_tolerance
        are counted as nonzero.

    Returns
    -------
    diagnostics : dict
    """
    singular_values = np.linalg.svd(
        D,
        compute_uv=False,
    )

    if singular_values.size == 0:
        return {
            "rank": 0,
            "required_rank": D.shape[1],
            "condition_number": np.inf,
            "relative_sigma_min": 0.0,
            "singular_values": singular_values,
            "threshold": np.nan,
        }

    sigma_max = singular_values[0]
    threshold = relative_tolerance * sigma_max

    numerical_rank = int(
        np.sum(singular_values > threshold)
    )

    required_rank = D.shape[1]

    if numerical_rank == required_rank:
        sigma_min = singular_values[required_rank - 1]
        condition_number = sigma_max / sigma_min
        relative_sigma_min = sigma_min / sigma_max
    else:
        condition_number = np.inf
        relative_sigma_min = (
            singular_values[-1] / sigma_max
        )

    return {
        "rank": numerical_rank,
        "required_rank": required_rank,
        "condition_number": condition_number,
        "relative_sigma_min": relative_sigma_min,
        "singular_values": singular_values,
        "threshold": threshold,
    }

info = numerical_rank_diagnostics(
    D,
    relative_tolerance=1e-12,
)

print("Number of rows:", D.shape[0])
print("Number of columns:", D.shape[1])
print("Required full rank:", info["required_rank"])
print("Numerical rank:", info["rank"])
print("Condition number:", info["condition_number"])
print(
    "sigma_min / sigma_max:",
    info["relative_sigma_min"],
)














for r in range(1, 6):
    filename = (
        "./burgers/exact_intrusive_operator/"
        f"exact_operator_reproj_noise{noise}_{r}.npz"
    )

    data = np.load(filename)
    Q_reproj = data["Q_reproj"]

    D, Phi, pairs = build_opinf_design_matrix(
        Q_reproj
    )

    info = numerical_rank_diagnostics(
        D,
        relative_tolerance=1e-12,
    )

    independent_features = (
        r + r * (r + 1) // 2
    )

    print("=" * 60)
    print(f"Reduced dimension r = {r}")
    print(f"Transitions          = {D.shape[0]}")
    print(f"Independent features = {independent_features}")
    print(f"Design matrix shape  = {D.shape}")
    print(f"Numerical rank       = {info['rank']}")
    print(
        f"Condition number     = "
        f"{info['condition_number']:.6e}"
    )
    print(
        f"sigma_min/sigma_max  = "
        f"{info['relative_sigma_min']:.6e}"
    )
    
    
    
    
    
    
# for r in [1,2,3,4,5]:
#     for k in range(995):
#         filename = (
#             "./burgers/exact_intrusive_operator/"
#             f"exact_operator_reproj_noise{noise}_{r}.npz"
#         )
    
#         data = np.load(filename)
#         Q_reproj = data["Q_reproj"]
    
#         D, Phi, pairs = build_opinf_design_matrix(
#             Q_reproj[:,:k]
#         )
    
#         info = numerical_rank_diagnostics(
#             D,
#             relative_tolerance=1e-12,
#         )
    
#         independent_features = (
#             r + r * (r + 1) // 2
#         )
        
#         if independent_features==info['rank']:
#             print(k)
#             break