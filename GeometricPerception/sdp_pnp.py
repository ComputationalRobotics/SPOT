from utils import generate_random_pnp_problem
from utils import project_to_SO3, rotation_error_degrees
import numpy as np
import sys 
import os
# Add the parent directory to sys.path so Python can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
import sympy as sp
import matplotlib.pyplot as plt

# generate a random PnP problem
N = 20
pnp = generate_random_pnp_problem(N=N, noise_std=0.1)
# Convert 2D noisy projections to bearing (unit-norm direction) vectors in camera frame
points_2d_noisy = pnp["points_2d_noisy"]  # shape: (N, 2)
# Homogenize (append 1) to make (N, 3)
bearing_vecs = np.hstack([points_2d_noisy, np.ones((points_2d_noisy.shape[0], 1))])
# Normalize each vector to unit length
bearing_vecs /= np.linalg.norm(bearing_vecs, axis=1, keepdims=True)

# define the polynomial optimization problem (POP) by symbolic variables
# define optimization variables
x = sp.symbols('x0:12')
R = sp.Matrix(3, 3, x[:9])
t = sp.Matrix(x[9:12])

# define objective function
objective = 0
for i in range(N):
    p3d = pnp['points_3d_world'][i]
    p3d_sym = sp.Matrix(p3d)
    transformed_p = R * p3d_sym + t
    bearing = bearing_vecs[i]
    # Compute Q = I - outer(bearing, bearing)
    I = np.eye(3)
    Q = I - np.outer(bearing, bearing)
    # The i-th objective is transformed_p.T @ Q @ transformed_p
    # Q is numpy, transformed_p is sympy, convert Q to sympy
    # this is the point to line distance squared
    Q_sym = sp.Matrix(Q)
    objective += (transformed_p.T * Q_sym * transformed_p)[0]

# define equality constraints
h = []
# 1. Each column of R inner product itself - 1 (should be unit vectors)
for i in range(3):
    h.append(R.col(i).dot(R.col(i)) - 1)

# 2. Each pair of columns inner product each other (should be orthogonal)
for i in range(3):
    for j in range(i+1, 3):
        h.append(R.col(i).dot(R.col(j)))

# 3. Cross product constraints for SO(3):
# (a) Cross product of column 1 and column 2 minus column 3 (should be zero vector)
cross_col12 = R.col(0).cross(R.col(1)) - R.col(2)
for i in range(3):
    h.append(cross_col12[i])
# (b) Cross product of column 2 and column 3 minus column 1 (should be zero vector)
cross_col23 = R.col(1).cross(R.col(2)) - R.col(0)
for i in range(3):
    h.append(cross_col23[i])
# (c) Cross product of column 3 and column 1 minus column 2 (should be zero vector)
cross_col31 = R.col(2).cross(R.col(0)) - R.col(1)
for i in range(3):
    h.append(cross_col31[i])

# define inequality constraints
# assume translation norm is less than 10
g = [100 - t.dot(t)]

# Call SPOT to formulate and solve the SDP relaxation of the POP
# set the parameters for SPOT (default, no need to change unless you want sparsity)
relax_mode = 'MOMENT' # Moment relaxation (versus SOS)
cs_mode = 'NON' # No CS (correlative sparsity)
ts_mode = 'NON' # No TS (term sparsity)
ts_mom_mode = 'NON' 
ts_eq_mode = 'NON' 
cs_cliques = [] # No CS cliques
if_solve = True # Solve the SDP
relaxtion_order = 1 # Shor's relaxation (higher order is Moment--SOS)
input_info = dict()
input_info['relax_mode'] = relax_mode 
input_info['cs_mode'] = cs_mode 
input_info['ts_mode'] = ts_mode 
input_info['ts_mom_mode'] = ts_mom_mode 
input_info['ts_eq_mode'] = ts_eq_mode 
input_info['if_solve'] = if_solve 
input_info['cliques'] = cs_cliques
# call SPOT python interface
result, res, coeff_info, aux_info = CSTSS_pybind(objective, g, h, relaxtion_order, x, input_info)

# extract the moment matrix
Xopt = res['Xopt'][0]

# extract optimal R and t
# Take the first column of Xopt and extract entries 1 to 9 (Python 1-based: 1:10, 0-based: 1:10)
col1 = Xopt[1:10, 0]
R_est = col1.reshape((3, 3))
R_est = project_to_SO3(R_est)
t_est = Xopt[10:, 0]

# Compute the error between the estimated R and t and the ground truth R and t
error_R = rotation_error_degrees(R_est, pnp['R_gt'])
error_t = np.linalg.norm(t_est - pnp['t_gt'])

print(f"Rotation error between R_mat and ground truth (degrees): {error_R:.4f}")
print(f"Translation error between t_est and t_gt: {error_t:.4f}")

# Compute the suboptimality gap
# Concatenate vectorization of R_est and t_est to form x_est
x_est = np.concatenate([R_est.flatten(), t_est])
# Evaluate the objective function at x_est by replacing symbolic variables x with x_est
upper_bound = objective.subs({var: val for var, val in zip(x, x_est)})
lower_bound = result # SDP optimal value is the lower bound
optimality_gap = abs(upper_bound - lower_bound) / (1 + abs(upper_bound) + abs(lower_bound))
print(f"Upper bound: {upper_bound:.6f}")
print(f"Lower bound: {lower_bound:.6f}")
print(f"Optimality gap: {optimality_gap:.2e}")










