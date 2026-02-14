from utils import generate_random_pnp_problem
from utils import project_to_SO3, rotation_error_degrees
import numpy as np
import sys
import os
import time
# Add the parent directory to sys.path so Python can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
from SPOT.PYTHON.numpoly import NumPolySystem, NumPolyExpr


def dot3(u, v):
    """Dot product of two length-3 lists of NumPolyExpr."""
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]


def cross3(u, v):
    """Cross product of two length-3 lists of NumPolyExpr."""
    return [
        u[1]*v[2] - u[2]*v[1],
        u[2]*v[0] - u[0]*v[2],
        u[0]*v[1] - u[1]*v[0]
    ]


def matvec3(M, v):
    """Matrix-vector product: M (3x3 list-of-lists) * v (length-3 list)."""
    return [dot3(M[i], v) for i in range(3)]


# generate a random PnP problem
N = 20
pnp = generate_random_pnp_problem(N=N, noise_std=0.1)
# Convert 2D noisy projections to bearing (unit-norm direction) vectors in camera frame
points_2d_noisy = pnp["points_2d_noisy"]  # shape: (N, 2)
# Homogenize (append 1) to make (N, 3)
bearing_vecs = np.hstack([points_2d_noisy, np.ones((points_2d_noisy.shape[0], 1))])
# Normalize each vector to unit length
bearing_vecs /= np.linalg.norm(bearing_vecs, axis=1, keepdims=True)

# define the polynomial optimization problem (POP)
# define optimization variables
n_vars = 12  # 9 for R + 3 for t
relaxation_order = 2  # Shor's relaxation

print(f"\nBuilding POP with NumPolySystem (n_vars={n_vars}, N={N})...")
t_build_start = time.time()

ps = NumPolySystem(n_vars=n_vars)
# x0..x8 = R (row-major: R[i][j] = x[3*i + j], same as sp.Matrix(3,3,x[:9]))
# x9,x10,x11 = t
x = [ps.var(i) for i in range(n_vars)]
R = [[x[3*i + j] for j in range(3)] for i in range(3)]  # R[i][j] = x[3*i + j]
t = [x[9], x[10], x[11]]
print("n_vars: ", n_vars)

# R columns: col(j) = [R[0][j], R[1][j], R[2][j]] = [x[j], x[3+j], x[6+j]]
R_col = [[R[i][j] for i in range(3)] for j in range(3)]

# define objective function
objective = NumPolyExpr.from_const(0)
for i in range(N):
    p3d = pnp['points_3d_world'][i]
    p3d_npe = [NumPolyExpr.from_const(float(p3d[j])) for j in range(3)]
    # transformed_p = R * p3d + t
    transformed_p = [dot3(R[k], p3d_npe) + t[k] for k in range(3)]
    bearing = bearing_vecs[i]
    # Compute Q = I - outer(bearing, bearing)
    I = np.eye(3)
    Q = I - np.outer(bearing, bearing)
    # The i-th objective is transformed_p.T @ Q @ transformed_p
    # this is the point to line distance squared
    for r in range(3):
        for c in range(3):
            if abs(Q[r, c]) > 1e-15:
                objective = objective + Q[r, c] * transformed_p[r] * transformed_p[c]

ps.set_obj(objective)

# define equality constraints
# 1. Each column of R inner product itself - 1 (should be unit vectors)
for i in range(3):
    ps.add_eq(dot3(R_col[i], R_col[i]) - 1.0)

# 2. Each pair of columns inner product each other (should be orthogonal)
for i in range(3):
    for j in range(i+1, 3):
        ps.add_eq(dot3(R_col[i], R_col[j]))

# 3. Cross product constraints for SO(3):
# (a) Cross product of column 0 and column 1 minus column 2 (should be zero vector)
cross_col01 = cross3(R_col[0], R_col[1])
for i in range(3):
    ps.add_eq(cross_col01[i] - R_col[2][i])
# (b) Cross product of column 1 and column 2 minus column 0 (should be zero vector)
cross_col12 = cross3(R_col[1], R_col[2])
for i in range(3):
    ps.add_eq(cross_col12[i] - R_col[0][i])
# (c) Cross product of column 2 and column 0 minus column 1 (should be zero vector)
cross_col20 = cross3(R_col[2], R_col[0])
for i in range(3):
    ps.add_eq(cross_col20[i] - R_col[1][i])

# define inequality constraints
# assume translation norm is less than 10
ps.add_ineq(NumPolyExpr.from_const(100.0) - dot3(t, t))

t_build = time.time() - t_build_start
print(f"POP construction time: {t_build:.3f}s")

# Clean polynomials
print("Cleaning polynomials...")
t_clean_start = time.time()
ps.clean_all(tol=1e-14, if_scale=True, scale_obj=False)
t_clean = time.time() - t_clean_start
print(f"Clean time: {t_clean:.3f}s")

# Convert to supp_rpt format
print("Converting to supp_rpt format...")
t_convert_start = time.time()
poly_data = ps.get_supp_rpt_data(kappa=relaxation_order)
t_convert = time.time() - t_convert_start
print(f"Convert time: {t_convert:.3f}s")

# Call SPOT to formulate and solve the SDP relaxation of the POP
# set the parameters for SPOT (default, no need to change unless you want sparsity)
input_info = dict()
input_info['relax_mode'] = 'MOMENT'  # Moment relaxation (versus SOS)
input_info['cs_mode'] = 'NON'       # No CS (correlative sparsity)
input_info['ts_mode'] = 'NON'       # No TS (term sparsity)
input_info['ts_mom_mode'] = 'NON'
input_info['ts_eq_mode'] = 'NON'
input_info['if_solve'] = True        # Solve the SDP
input_info['cliques'] = []           # No CS cliques
# call SPOT python interface
result, res, coeff_info, aux_info = CSTSS_pybind(poly_data, relaxation_order, n_vars, input_info)

# extract the moment matrix
Xopt = res['Xopt'][0]

# extract optimal R and t
# Take the first column of Xopt and extract entries 1 to 9 (Python 1-based: 1:10, 0-based: 1:10)
col1 = Xopt[1:10, 0]
R_est = col1.reshape((3, 3))
R_est = project_to_SO3(R_est)
t_est = Xopt[10:13, 0]

# Compute the error between the estimated R and t and the ground truth R and t
error_R = rotation_error_degrees(R_est, pnp['R_gt'])
error_t = np.linalg.norm(t_est - pnp['t_gt'])

print(f"Rotation error between R_mat and ground truth (degrees): {error_R:.4f}")
print(f"Translation error between t_est and t_gt: {error_t:.4f}")

# Compute the suboptimality gap
# Evaluate the objective function numerically at (R_est, t_est)
upper_bound = 0.0
for i in range(N):
    p3d = pnp['points_3d_world'][i]
    transformed_p = R_est @ p3d + t_est
    bearing = bearing_vecs[i]
    Q = np.eye(3) - np.outer(bearing, bearing)
    upper_bound += transformed_p @ Q @ transformed_p

lower_bound = result # SDP optimal value is the lower bound
optimality_gap = abs(upper_bound - lower_bound) / (1 + abs(upper_bound) + abs(lower_bound))
print(f"Upper bound: {upper_bound:.6f}")
print(f"Lower bound: {lower_bound:.6f}")
print(f"Optimality gap: {optimality_gap:.2e}")
