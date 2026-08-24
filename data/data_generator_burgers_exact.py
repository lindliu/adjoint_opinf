

import numpy as np
from scipy.sparse import diags
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Parameters
# ------------------------------------------------------------
N = 128
L = 1.0
nu = 0.01
T = 1.0

dx = L / (N + 1)
x = np.linspace(0, L, N + 2)

q0 = np.sin(2 * np.pi * x[1:-1])

# Stable time step for explicit RK4
dt_diff = 0.2 * dx**2 / nu
dt_conv = 0.2 * dx / max(np.max(np.abs(q0)), 1e-12)

dt = min(dt_diff, dt_conv)
M = int(np.ceil(T / dt))
M = 999 if M<999 else M
dt = T / M

t = np.linspace(0, T, M + 1)

print(f"N = {N}, dx = {dx:.3e}, dt = {dt:.3e}, M = {M}")


# ------------------------------------------------------------
# FD matrices
# ------------------------------------------------------------
D1 = diags(
    [
        -np.ones(N - 1) / (2.0 * dx),
        np.ones(N - 1) / (2.0 * dx),
    ],
    offsets=[-1, 1],
    shape=(N, N),
).tocsc()  #.toarray()

D2 = diags(
    [
        np.ones(N - 1) / dx**2,
        -2.0 * np.ones(N) / dx**2,
        np.ones(N - 1) / dx**2,
    ],
    offsets=[-1, 0, 1],
    shape=(N, N),
).tocsc()


def add_noise(Q, percentage, method="max_norm"):
    """
    Add Gaussian noise to a snapshot matrix Q.
    
    Parameters:
        Q (np.ndarray): Input matrix with shape (state_dimension, num_snapshots)
        percentage (float): Noise percentage (e.g., 5 for 5% noise)
        method (str): 
            - "max_norm": Noise scaled by max column norm of Q
            - "std": Noise scaled by standard deviation of Q
    
    Returns:
        Q_noisy (np.ndarray): Noisy version of Q
    """
    
    # Validate input
    if percentage < 0:
        raise ValueError("Percentage must be between 0 and 100")
    
    if method == "max_norm":
        # Calculate maximum column norm
        norms = np.linalg.norm(Q, axis=0)
        scale = np.max(norms) * (percentage / 100)
    elif method == "std":
        # Calculate standard deviation of entire matrix
        scale = np.std(Q) * (percentage / 100)
    else:
        raise ValueError("Invalid method. Choose 'max_norm' or 'std'")
    
    # Generate Gaussian noise
    noise = np.random.normal(loc=0, scale=scale, size=Q.shape)
    
    # Add noise to original data
    Q_noisy = Q + noise
    
    return Q_noisy
# ------------------------------------------------------------
# Continuous-time Burgers RHS, conservative form
# q_t = nu q_xx - 0.5 (q^2)_x
# ------------------------------------------------------------
def burgers_rhs(q):
    return nu * (D2 @ q) - 0.5 * (D1 @ (q**2))


def euler_step(q, dt):
    return q + dt * burgers_rhs(q)

def rk4_step(q, dt):
    k1 = burgers_rhs(q)
    k2 = burgers_rhs(q + 0.5 * dt * k1)
    k3 = burgers_rhs(q + 0.5 * dt * k2)
    k4 = burgers_rhs(q + dt * k3)

    return q + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0


# ------------------------------------------------------------
# Generate FOM trajectory
# ------------------------------------------------------------
Q = np.zeros((N, M + 1))
Q[:, 0] = q0

q = q0.copy()

for k in range(M):
    q = rk4_step(q, dt)
    # q = euler_step(q, dt)
    
    if not np.all(np.isfinite(q)):
        raise RuntimeError(f"Solution blew up at step {k}")

    Q[:, k + 1] = q



# ------------------------------------------------------------
# Re-projected clean reduced data
# ------------------------------------------------------------
def generate_reprojected_data(q0, V, M):
    r = V.shape[1]

    Q_reproj = np.zeros((r, M + 1))

    qhat = V.T @ q0
    Q_reproj[:, 0] = qhat

    for k in range(M):
        q_lift = V @ qhat
        # q_next = rk4_step(q_lift, dt)
        q_next = euler_step(q_lift, dt)
        
        
        
        qhat = V.T @ q_next

        if not np.all(np.isfinite(qhat)):
            raise RuntimeError(f"Re-projected trajectory blew up at step {k}")

        Q_reproj[:, k + 1] = qhat

    return Q_reproj

# ------------------------------------------------------------
# Intrusive ROM operators
# ------------------------------------------------------------
def build_intrusive_burgers_operators(V):
    r = V.shape[1]

    A_r = V.T @ ((nu * D2) @ V)

    H_tensor = np.zeros((r, r, r))

    for i in range(r):
        vi = V[:, i]
        for j in range(r):
            vj = V[:, j]

            nonlinear_ij = -0.5 * (D1 @ (vi * vj))
            H_tensor[:, i, j] = V.T @ nonlinear_ij

    H_mat = H_tensor.reshape(r, r * r)

    return A_r, H_mat, H_tensor
    
    
for r in range(1,6):
    # ------------------------------------------------------------
    # POD basis
    # ------------------------------------------------------------
    # r = 5
    
    U, S, VT = np.linalg.svd(Q, full_matrices=False)
    V = U[:, :r]
    
    Q_proj = V.T @ Q
    
    # Re-projected clean reduced data
    Q_reproj = generate_reprojected_data(q0, V, M)

    # Intrusive ROM operators
    A_r_true, H_r_true, H_tensor_true = build_intrusive_burgers_operators(V)
    
    
    np.savez(f'./burgers/exact_intrusive_operator/exact_operator_reproj_noise{0}_{r}.npz', \
             A_r_true=A_r_true, H_r_true=H_r_true, Q_reproj=Q_reproj, t=t)

    noise_level = 20
    Q_reproj_noised = add_noise(Q_reproj, percentage=noise_level, method="std")
    np.savez(f'./burgers/exact_intrusive_operator/exact_operator_reproj_noise{noise_level}_{r}.npz', \
             A_r_true=A_r_true, H_r_true=H_r_true, Q_reproj=Q_reproj_noised, t=t)
        
# plt.plot(Q_proj.T)
# plt.plot(Q_reproj.T)


# ------------------------------------------------------------
# Consistency check
# ------------------------------------------------------------
def rom_rhs(qhat, A_r, H_r):
    return A_r @ qhat + H_r @ np.kron(qhat, qhat)


qhat_test = Q_reproj[:, 0]
q_lift = V @ qhat_test

rhs_projected = V.T @ burgers_rhs(q_lift)
rhs_rom = rom_rhs(qhat_test, A_r_true, H_r_true)

print("ROM operator consistency error:")
print(np.linalg.norm(rhs_projected - rhs_rom))