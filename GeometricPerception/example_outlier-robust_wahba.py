from utils import generate_random_wahba_problem, solve_wahba_svd
from utils import project_to_SO3, rotation_error_degrees
import numpy as np
import sys 
import os
# Add the parent directory to sys.path so Python can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
import sympy as sp
import matplotlib.pyplot as plt

# define the rotation matrix from quaternion
def Rofq(q):
    q1, q2, q3, q4 = q
    return sp.Matrix([
        [q1**2 - q2**2 - q3**2 + q4**2, 2*(q1*q2 - q3*q4), 2*(q1*q3 + q2*q4)],
        [2*(q1*q2 + q3*q4), -q1**2 + q2**2 - q3**2 + q4**2, 2*(q2*q3 - q1*q4)],
        [2*(q1*q3 - q2*q4), 2*(q2*q3 + q1*q4), -q1**2 - q2**2 + q3**2 + q4**2]
    ])
# define the quaternion from rotation matrix
def qofR(R):
    """
    Given a 3x3 rotation matrix R (numpy array), return a quaternion q
    such that Rofq(q) = R, using the convention in Rofq.
    Returns q as a length-4 numpy array (q1, q2, q3, q4).
    """
    # Ensure R is a numpy array
    R = np.asarray(R)
    # The following is adapted to match Rofq ordering: q1,q2,q3,q4
    # Algorithm: eigenvector of the quaternion matrix (Bar-Itzhack 2000), adapted to match Rofq convention
    K = np.zeros((4,4))
    K[0,0] = R[0,0] - R[1,1] - R[2,2]
    K[0,1] = R[0,1] + R[1,0]
    K[0,2] = R[0,2] + R[2,0]
    K[0,3] = R[1,2] - R[2,1]

    K[1,0] = K[0,1]
    K[1,1] = -R[0,0] + R[1,1] - R[2,2]
    K[1,2] = R[1,2] + R[2,1]
    K[1,3] = R[2,0] - R[0,2]

    K[2,0] = K[0,2]
    K[2,1] = K[1,2]
    K[2,2] = -R[0,0] - R[1,1] + R[2,2]
    K[2,3] = R[0,1] - R[1,0]

    K[3,0] = K[0,3]
    K[3,1] = K[1,3]
    K[3,2] = K[2,3]
    K[3,3] = R[0,0] + R[1,1] + R[2,2]

    # Normalize by 3
    K = K / 3.0
    # Compute largest eigenvalue/eigenvector
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[:, np.argmax(eigvals)] # The eigenvector of largest eigenvalue

    # The sign of q is ambiguous. Ensure q4 >= 0 (q4 is scalar in our convention)
    if q[3] < 0:
        q = -q

    # The K matrix algorithm gives the conjugate for our Rofq convention
    # Conjugate: negate the vector part (q1, q2, q3), keep scalar (q4)
    q[:3] = -q[:3]

    return q

# generate a random Wahba problem
N = 100
outlier_ratio = 0.6
wahba = generate_random_wahba_problem(N=N, noise_std=0.02, outlier_ratio=outlier_ratio)
# Extract unit vectors a and b
a = wahba['a']
b = wahba['b']
R_gt = wahba['R_gt']
# Use R_gt to transform vectors in a, then do b - R*a, and compute the squared errors
a_transformed = (R_gt @ a.T).T  # (N, 3) matrix: apply R_gt to each vector in a
errors = b - a_transformed      # (N, 3) residuals
sq_errors = np.sum(errors**2, axis=1)  # Squared error for each correspondence
print("Squared errors for each correspondence:", sq_errors)
outlier_indices = np.where(wahba['outlier_mask'])[0]
print("Indices of ground truth outliers:", outlier_indices)

# Define the polynomial optimization problem (POP) by symbolic variables
beta = 0.5 # noise bound, tune this parameter according to noise_std
n = 4 + N # total number of variables
v = sp.symbols(f'v0:{n}') # define the variables
q = v[:4] # quaternion
th = v[4:] # binary variable for outlier detection: 1 for inlier, -1 for outlier
q_vec = sp.Matrix(q) # quaternion vector
th_vec = sp.Matrix(th) # binary variable vector
Rq = Rofq(q) # rotation matrix from quaternion
# define the objective function
# according to Yang, Heng, and Luca Carlone. "A quaternion-based certifiably optimal solution to the Wahba problem with outliers." In ICCV 2019.
f = 0
for i in range(N):
    a_i = sp.Matrix(a[i,:])
    b_i = sp.Matrix(b[i,:])
    th_i = th[i]
    term = 1.0 * ((1 + th_i)/2) * (b_i.norm()**2 + a_i.norm()**2 - 2 * b_i.dot(Rq * a_i)) + ((1 - th_i)/2) * beta**2
    f += term
# define the equality constraints
h = [1.0 - q_vec.dot(q_vec)] # quaternion constraint
for i in range(N):
    h.append(1.0 - th[i]**2) # binary variable constraint
# define the inequality constraints
g = [] # no inequality constraints

# call SPOT to formulate and solve the SDP relaxation of the POP
input_info = dict()
input_info['cs_mode'] = 'MF' # consider correlative sparsity: more scalable than the relaxation in Yang et al. 2019
input_info['ts_mode'] = 'NON' # no term sparsity
input_info['relax_mode'] = 'SOS' # SOS relaxation (faster to solve for MOSEK than Moment relaxation )
input_info['cliques'] = [] # let cs_mode decide the cliques
input_info['ts_mom_mode'] = 'NON'
input_info['ts_eq_mode'] = 'NON'
input_info['if_solve'] = True
relaxtion_order = 2 # relaxation order: 2 or 3 are both good choices,we choose 3 to be tighter
result, res, coeff_info, aux_info = CSTSS_pybind(f, g, h, relaxtion_order, v, input_info)

# extract the moment matrix
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

# Find estimated outlier indices by inspecting th_est_raw.
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

# Get inlier indices (indices not in outlier_indices_est)
all_indices = set(range(a.shape[0]))
inlier_indices_est = [idx for idx in all_indices if idx not in outlier_indices_est]

# Extract inlier correspondences for a and b based on estimated inliers
a_inlier_est = a[inlier_indices_est]
b_inlier_est = b[inlier_indices_est]
R_est = solve_wahba_svd(a_inlier_est, b_inlier_est)
error_R = rotation_error_degrees(R_est, R_gt)
print("Rotation error between estimated R and ground truth R (degrees):", error_R)

# evaluate the objective function at the estimated [q; th] (upper bound)
q_est = qofR(R_est)
v_est = np.concatenate([q_est, th_est])
f_numeric = f.subs({vi: float(v_est[i]) for i, vi in enumerate(v)})
f_evaluated = float(f_numeric)
upper_bound = f_evaluated
lower_bound = result 
optimality_gap = abs(upper_bound - lower_bound) / (1 + abs(upper_bound) + abs(lower_bound))
print(f"Upper bound: {upper_bound:.6f}")
print(f"Lower bound: {lower_bound:.6f}")
print(f"Optimality gap: {optimality_gap:.2e}")












