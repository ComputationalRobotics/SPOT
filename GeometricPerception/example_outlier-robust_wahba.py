from utils import generate_random_wahba_problem, solve_wahba_svd
from utils import project_to_SO3, rotation_error_degrees
import numpy as np
import sys
import os
import time

# Add the parent directory to sys.path so Python can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
from SPOT.PYTHON.numpoly import NumPolySystem, NumPolyExpr


# define the rotation matrix from quaternion using NumPolyExpr
def Rofq_numpoly(q):
    """Build R(q) as a 3x3 list-of-lists of NumPolyExpr from quaternion variables q0..q3."""
    q1, q2, q3, q4 = q
    return [
        [q1**2 - q2**2 - q3**2 + q4**2, 2*(q1*q2 - q3*q4), 2*(q1*q3 + q2*q4)],
        [2*(q1*q2 + q3*q4), -q1**2 + q2**2 - q3**2 + q4**2, 2*(q2*q3 - q1*q4)],
        [2*(q1*q3 - q2*q4), 2*(q2*q3 + q1*q4), -q1**2 - q2**2 + q3**2 + q4**2]
    ]


def dot3(u, v):
    """Dot product of two length-3 lists of NumPolyExpr."""
    return u[0]*v[0] + u[1]*v[1] + u[2]*v[2]


def norm2_3(u):
    """Squared norm of a length-3 list of NumPolyExpr."""
    return dot3(u, u)


def matvec3(M, v):
    """Matrix-vector product: M (3x3 list-of-lists) * v (length-3 list)."""
    return [dot3(M[i], v) for i in range(3)]


# define the quaternion from rotation matrix (numerical, same as original)
def qofR(R):
    R = np.asarray(R)
    K = np.zeros((4, 4))
    K[0, 0] = R[0, 0] - R[1, 1] - R[2, 2]
    K[0, 1] = R[0, 1] + R[1, 0]
    K[0, 2] = R[0, 2] + R[2, 0]
    K[0, 3] = R[1, 2] - R[2, 1]
    K[1, 0] = K[0, 1]
    K[1, 1] = -R[0, 0] + R[1, 1] - R[2, 2]
    K[1, 2] = R[1, 2] + R[2, 1]
    K[1, 3] = R[2, 0] - R[0, 2]
    K[2, 0] = K[0, 2]
    K[2, 1] = K[1, 2]
    K[2, 2] = -R[0, 0] - R[1, 1] + R[2, 2]
    K[2, 3] = R[0, 1] - R[1, 0]
    K[3, 0] = K[0, 3]
    K[3, 1] = K[1, 3]
    K[3, 2] = K[2, 3]
    K[3, 3] = R[0, 0] + R[1, 1] + R[2, 2]
    K = K / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)]
    if q[3] < 0:
        q = -q
    q[:3] = -q[:3]
    return q


# generate a random Wahba problem
np.random.seed(42)
N = 100
outlier_ratio = 0.8
wahba = generate_random_wahba_problem(N=N, noise_std=0.02, outlier_ratio=outlier_ratio)
a = wahba['a']
b = wahba['b']
R_gt = wahba['R_gt']
a_transformed = (R_gt @ a.T).T
errors = b - a_transformed
sq_errors = np.sum(errors**2, axis=1)
print("Squared errors for each correspondence:", sq_errors)
outlier_indices = np.where(wahba['outlier_mask'])[0]
print("Indices of ground truth outliers:", outlier_indices)

# Build the polynomial optimization problem using NumPolySystem (no SymPy!)
beta = 0.5  # noise bound
n = 4 + N   # total number of variables: 4 quaternion + N binary
relaxation_order = 3


print(f"\nBuilding POP with NumPolySystem (n_vars={n}, N={N})...")
t_build_start = time.time()

ps = NumPolySystem(n_vars=n)

# variables: v0..v3 = quaternion, v4..v(4+N-1) = binary outlier indicators
q = [ps.var(i) for i in range(4)]
th = [ps.var(4 + i) for i in range(N)]

# Build R(q)
Rq = Rofq_numpoly(q)

# Build the objective function
# f = sum_i [ (1+th_i)/2 * (||b_i||^2 + ||a_i||^2 - 2*b_i'*R(q)*a_i) + (1-th_i)/2 * beta^2 ]
f = NumPolyExpr.from_const(0)
for i in range(N):
    a_i = [NumPolyExpr.from_const(float(a[i, j])) for j in range(3)]
    b_i = [NumPolyExpr.from_const(float(b[i, j])) for j in range(3)]
    th_i = th[i]

    # R(q) * a_i
    Rq_ai = matvec3(Rq, a_i)

    # ||b_i||^2 + ||a_i||^2 (these are scalars since a_i, b_i are unit vectors ~ 1.0 each)
    b_norm2 = float(np.sum(b[i, :]**2))
    a_norm2 = float(np.sum(a[i, :]**2))

    # b_i' * R(q) * a_i
    b_dot_Rqa = dot3(b_i, Rq_ai)

    # residual = ||b_i||^2 + ||a_i||^2 - 2*b_i'*R(q)*a_i
    residual = NumPolyExpr.from_const(b_norm2 + a_norm2) - 2.0 * b_dot_Rqa

    # (1+th_i)/2 * residual + (1-th_i)/2 * beta^2
    term = 0.5 * (1.0 + th_i) * residual + 0.5 * (1.0 - th_i) * NumPolyExpr.from_const(beta**2)
    f = f + term

ps.set_obj(f)

# Equality constraints
# 1) quaternion unit norm: 1 - q'*q = 0
q_norm_sq = q[0]**2 + q[1]**2 + q[2]**2 + q[3]**2
ps.add_eq(NumPolyExpr.from_const(1.0) - q_norm_sq)

# 2) binary constraints: 1 - th_i^2 = 0 for each i
for i in range(N):
    ps.add_eq(NumPolyExpr.from_const(1.0) - th[i]**2)

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

# Call CSTSS optimized solver
input_info = dict()
input_info['cs_mode'] = 'MF'
input_info['ts_mode'] = 'NON'
input_info['relax_mode'] = 'SOS'
input_info['cliques'] = []
input_info['ts_mom_mode'] = 'NON'
input_info['ts_eq_mode'] = 'NON'
input_info['if_solve'] = True

result, res, coeff_info, aux_info = CSTSS_pybind(
    poly_data, relaxation_order, n, input_info
)

# Extract the moment matrix
if input_info['relax_mode'] == 'MOMENT':
    Xs = res['Xopt']
elif input_info['relax_mode'] == 'SOS':
    Xs = []
    for S in res['Sopt']:
        Xs.append(-S)

# Extract the binary variables th_est:
th_est_raw = []
for X in Xs:
    th_est_raw.append(X[5, 0])
th_est_raw = np.array(th_est_raw)
print("Estimated binary variables (th_est_raw):", th_est_raw)

# Find estimated outlier indices by inspecting th_est_raw
outlier_indices_est = []
for idx, th_val in enumerate(th_est_raw):
    if th_val < 0:
        clique = aux_info['cliques'][idx]
        outlier_index = clique[-1] - 5  # adjust for Python's 0-based indexing
        outlier_indices_est.append(outlier_index)
print("Estimated outlier indices (0-based):", outlier_indices_est)
print("ground truth outlier indices (0-based):", outlier_indices)

# Create a new th_est: -1 at outlier_indices_est, +1 elsewhere
th_est = np.ones_like(th_est_raw)
for idx in outlier_indices_est:
    th_est[idx] = -1

# Get inlier indices
all_indices = set(range(a.shape[0]))
inlier_indices_est = [idx for idx in all_indices if idx not in outlier_indices_est]

# Extract inlier correspondences and solve
a_inlier_est = a[inlier_indices_est]
b_inlier_est = b[inlier_indices_est]
R_est = solve_wahba_svd(a_inlier_est, b_inlier_est)
error_R = rotation_error_degrees(R_est, R_gt)
print("Rotation error between estimated R and ground truth R (degrees):", error_R)

# Evaluate the objective function at the estimated [q; th] (upper bound)
# We evaluate numerically by substituting into the polynomial
q_est = qofR(R_est)

# Compute f(q_est, th_est) numerically (without SymPy)
upper_bound = 0.0
R_est_mat = R_est
for i in range(N):
    a_i = a[i, :]
    b_i = b[i, :]
    th_i = th_est[i]
    residual_i = np.sum(b_i**2) + np.sum(a_i**2) - 2.0 * b_i @ (R_est_mat @ a_i)
    upper_bound += 0.5 * (1.0 + th_i) * residual_i + 0.5 * (1.0 - th_i) * beta**2

lower_bound = result
optimality_gap = abs(upper_bound - lower_bound) / (1 + abs(upper_bound) + abs(lower_bound))
print(f"Upper bound: {upper_bound:.6f}")
print(f"Lower bound: {lower_bound:.6f}")
print(f"Optimality gap: {optimality_gap:.2e}")

