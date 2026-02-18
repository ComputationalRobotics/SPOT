import os
import datetime
import numpy as np
import pickle
import time

import sys
# Add the parent directory to sys.path so Python can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
from SPOT.PYTHON.numpoly import NumPolySystem, NumPolyExpr, numpoly_visualize
from SPOT.PYTHON.naive_extract import naive_extract
from SPOT.PYTHON.robust_extract_CS import robust_extract_CS, ordered_extract_CS

###############################
# Helper Functions and Stubs  #
###############################

def get_id(prefix, k, N):
    """
    Returns the variable id (1-indexed, like MATLAB) for a given prefix and stage k.
    """
    if prefix == "rc":
        return k + 1
    elif prefix == "rs":
        return (N + 1) + k + 1
    elif prefix == "sx":
        return 2 * N + 2 + k + 1
    elif prefix == "sy":
        return 3 * N + 3 + k + 1
    elif prefix == "fc":
        return 4 * N + 4 + k
    elif prefix == "fs":
        return 5 * N + 4 + k
    elif prefix == "px":
        return 6 * N + 4 + k
    elif prefix == "py":
        return 7 * N + 4 + k
    elif prefix == "Fx":
        return 8 * N + 4 + k
    elif prefix == "Fy":
        return 9 * N + 4 + k
    elif prefix == "lam1":
        return 10 * N + 4 + k
    elif prefix == "lam2":
        return 11 * N + 4 + k
    elif prefix == "lam3":
        return 12 * N + 4 + k
    elif prefix == "lam4":
        return 13 * N + 4 + k
    elif prefix == "lam5":
        return 14 * N + 4 + k
    elif prefix == "lam6":
        return 15 * N + 4 + k
    elif prefix == "lam7":
        return 16 * N + 4 + k
    elif prefix == "lam8":
        return 17 * N + 4 + k
    else:
        raise ValueError("Unknown prefix")


def get_var_mapping(params):
    """
    Creates a mapping (dictionary) from variable id to a string describing the variable.
    """
    id_func = params['id']
    N = params['N']
    var_mapping = {}
    # For states at time 0 to N
    for k in range(N + 1):
        var_mapping[id_func("rc", k)] = f"r_{{c, {k}}}"
        var_mapping[id_func("rs", k)] = f"r_{{s, {k}}}"
        var_mapping[id_func("sx", k)] = f"s_{{x, {k}}}"
        var_mapping[id_func("sy", k)] = f"s_{{y, {k}}}"
    for k in range(1, N + 1):
        var_mapping[id_func("fc", k)] = f"f_{{c, {k}}}"
        var_mapping[id_func("fs", k)] = f"f_{{s, {k}}}"
        var_mapping[id_func("px", k)] = f"p_{{x, {k}}}"
        var_mapping[id_func("py", k)] = f"p_{{y, {k}}}"
        var_mapping[id_func("Fx", k)] = f"F_{{x, {k}}}"
        var_mapping[id_func("Fy", k)] = f"F_{{y, {k}}}"
        var_mapping[id_func("lam1", k)] = f"\lambda_{{1, {k}}}"
        var_mapping[id_func("lam2", k)] = f"\lambda_{{2, {k}}}"
        var_mapping[id_func("lam3", k)] = f"\lambda_{{3, {k}}}"
        var_mapping[id_func("lam4", k)] = f"\lambda_{{4, {k}}}"
        var_mapping[id_func("lam5", k)] = f"\lambda_{{5, {k}}}"
        var_mapping[id_func("lam6", k)] = f"\lambda_{{6, {k}}}"
        var_mapping[id_func("lam7", k)] = f"\lambda_{{7, {k}}}"
        var_mapping[id_func("lam8", k)] = f"\lambda_{{8, {k}}}"
    return var_mapping


def get_remapped_ids(params):
    """
    Remaps variable ids to recover a chain‐like sparsity pattern.
    The MATLAB version uses 1-indexing; here we store results in a NumPy array.
    """
    N = params['N']
    total_var_num = params['total_var_num']
    id_func = params['id']
    ids_remap = np.zeros(total_var_num, dtype=int)
    idx = 1  # Maintaining MATLAB-style indexing (starting at 1)
    for k in range(N + 1):
        ids_remap[get_id("rc", k, N) - 1] = idx; idx += 1
        ids_remap[get_id("rs", k, N) - 1] = idx; idx += 1
        ids_remap[get_id("sx", k, N) - 1] = idx; idx += 1
        ids_remap[get_id("sy", k, N) - 1] = idx; idx += 1
        if k > 0:
            ids_remap[get_id("fc", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("fs", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("px", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("py", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("Fx", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("Fy", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("lam1", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("lam2", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("lam3", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("lam4", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("lam5", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("lam6", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("lam7", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("lam8", k, N) - 1] = idx; idx += 1
    return ids_remap


def get_eq_init(rc_0, rs_0, sx_0, sy_0, th_init, sx_init, sy_init,
                dt, m, g, mu1, mu2, c, l, dc, r, fc_min, eta):
    """
    Compute the initial state equality constraints.
    """
    eq1 = rc_0 - np.cos(th_init)
    eq2 = rs_0 - np.sin(th_init)
    eq3 = sx_0 - sx_init
    eq4 = sy_0 - sy_init
    eq5 = rc_0**2 + rs_0**2 - 1
    eqs = [eq1, eq2, eq3, eq4, eq5]
    eq_mask = [1, 1, 1, 1, 0]
    return eqs, eq_mask


def get_eq_dyn(rc_km1, rc_k, rs_km1, rs_k, sx_km1, sx_k, sy_km1, sy_k,
               fc_km1, fs_km1, px_km1, py_km1, Fx_km1, Fy_km1,
               lam1_km1, lam2_km1, lam3_km1, lam4_km1, lam5_km1, lam6_km1, lam7_km1, lam8_km1,
               dt, m, g, mu1, mu2, c, l, dc, r, fc_min, eta, if_sum_eq):
    """
    Compute the dynamic equality constraints.
    """
    eq1 = sx_k - sx_km1 - dt * (rc_km1 * Fx_km1 - rs_km1 * Fy_km1)
    eq2 = sy_k - sy_km1 - dt * (rs_km1 * Fx_km1 + rc_km1 * Fy_km1)
    eq3 = fs_km1 - dt/(c*r) * (-py_km1 * Fx_km1 + px_km1 * Fy_km1)
    eq4 = rc_k - (rc_km1 * fc_km1 - rs_km1 * fs_km1)
    eq5 = rs_k - (rs_km1 * fc_km1 + rc_km1 * fs_km1)
    eq6 = rc_k**2 + rs_k**2 - 1
    eq7 = fc_km1**2 + fs_km1**2 - 1

    # Geometry for contact modes
    x1 = -2.0 * l
    x2 = -0.5 * l
    x3 = 0.5 * l
    x4 = 2.0 * l
    y1 = -dc * l
    y2 = (3.0 - dc) * l
    y3 = (4.0 - dc) * l

    eq_con_mode = [
        lam1_km1 * (1.0 - lam1_km1),
        lam2_km1 * (1.0 - lam2_km1),
        lam3_km1 * (1.0 - lam3_km1),
        lam4_km1 * (1.0 - lam4_km1),
        lam5_km1 * (1.0 - lam5_km1),
        lam6_km1 * (1.0 - lam6_km1),
        lam7_km1 * (1.0 - lam7_km1),
        lam8_km1 * (1.0 - lam8_km1),
        lam1_km1**2 + lam2_km1**2 + lam3_km1**2 + lam4_km1**2 + \
          lam5_km1**2 + lam6_km1**2 + lam7_km1**2 + lam8_km1**2 - 1.0
    ]

    if if_sum_eq:
        eq_con_mode.extend([
            (lam1_km1 * (py_km1 - y3) +
             lam2_km1 * (px_km1 - x4) +
             lam3_km1 * (py_km1 - y2) +
             lam4_km1 * (px_km1 - x3) +
             lam5_km1 * (py_km1 - y1) +
             lam6_km1 * (px_km1 - x2) +
             lam7_km1 * (py_km1 - y2) +
             lam8_km1 * (px_km1 - x1)),
            (lam1_km1 * Fx_km1 +
             lam2_km1 * Fy_km1 +
             lam3_km1 * Fx_km1 +
             lam4_km1 * Fy_km1 +
             lam5_km1 * Fx_km1 +
             lam6_km1 * Fy_km1 +
             lam7_km1 * Fx_km1 +
             lam8_km1 * Fy_km1)
        ])
    else:
        eq_con_mode.extend([
            lam1_km1 * (py_km1 - y3),
            lam2_km1 * (px_km1 - x4),
            lam3_km1 * (py_km1 - y2),
            lam4_km1 * (px_km1 - x3),
            lam5_km1 * (py_km1 - y1),
            lam6_km1 * (px_km1 - x2),
            lam7_km1 * (py_km1 - y2),
            lam8_km1 * (px_km1 - x1),
            lam1_km1 * Fx_km1,
            lam2_km1 * Fy_km1,
            lam3_km1 * Fx_km1,
            lam4_km1 * Fy_km1,
            lam5_km1 * Fx_km1,
            lam6_km1 * Fy_km1,
            lam7_km1 * Fx_km1,
            lam8_km1 * Fy_km1
        ])
    # Define eq_mask: (1 indicates an equality to be enforced strictly)
    eq_mask = [1, 1, 1, 1, 1, 0, 1] + [1] * len(eq_con_mode)
    eqs = [eq1, eq2, eq3, eq4, eq5, eq6, eq7] + eq_con_mode
    return eqs, eq_mask


def enlarge(x, eta):
    """Increase x by eta."""
    return x + eta


def reduce_func(x, eta):
    """Decrease x by eta (avoid name conflict with built-in reduce)."""
    return x - eta


def get_ineq_dyn(rc_km1, rc_k, rs_km1, rs_k, sx_km1, sx_k, sy_km1, sy_k,
                 fc_km1, fs_km1, px_km1, py_km1, Fx_km1, Fy_km1,
                 lam1_km1, lam2_km1, lam3_km1, lam4_km1, lam5_km1, lam6_km1, lam7_km1, lam8_km1,
                 dt, m, g, mu1, mu2, c, l, dc, r, fc_min, eta, if_sum_ineq):
    """
    Compute the dynamic inequality constraints.
    """
    x1 = -2.0 * l; x2 = -0.5 * l; x3 = 0.5 * l; x4 = 2.0 * l
    y1 = -dc * l; y2 = (3.0 - dc) * l; y3 = (4.0 - dc) * l

    # Inequality for minimum contact force
    ineq1 = fc_km1 - fc_min

    if if_sum_ineq:
        ineq_con_mode = (lam1_km1 * (px_km1 - enlarge(x1, eta)) +
                         lam2_km1 * (py_km1 - enlarge(y2, eta)) +
                         lam3_km1 * (px_km1 - enlarge(x3, eta)) +
                         lam4_km1 * (py_km1 - enlarge(y1, eta)) +
                         lam5_km1 * (px_km1 - enlarge(x2, eta)) +
                         lam6_km1 * (py_km1 - enlarge(y1, eta)) +
                         lam7_km1 * (px_km1 - enlarge(x1, eta)) +
                         lam8_km1 * (py_km1 - enlarge(y2, eta)))
        ineq_con_mode_2 = (lam1_km1 * (reduce_func(x4, eta) - px_km1) +
                           lam2_km1 * (reduce_func(y3, eta) - py_km1) +
                           lam3_km1 * (reduce_func(x4, eta) - px_km1) +
                           lam4_km1 * (reduce_func(y2, eta) - py_km1) +
                           lam5_km1 * (reduce_func(x3, eta) - px_km1) +
                           lam6_km1 * (reduce_func(y2, eta) - py_km1) +
                           lam7_km1 * (reduce_func(x2, eta) - px_km1) +
                           lam8_km1 * (reduce_func(y3, eta) - py_km1))
        ineq_con_mode_3 = (lam1_km1 * (-Fy_km1) +
                           lam2_km1 * (-Fx_km1) +
                           lam3_km1 * (Fy_km1) +
                           lam4_km1 * (-Fx_km1) +
                           lam5_km1 * (Fy_km1) +
                           lam6_km1 * (Fx_km1) +
                           lam7_km1 * (Fy_km1) +
                           lam8_km1 * (Fx_km1))
        ineq = [ineq1, ineq_con_mode, ineq_con_mode_2, ineq_con_mode_3]
    else:
        ineq_con_mode = [
            lam1_km1 * (px_km1 - enlarge(x1, eta)),
            lam2_km1 * (py_km1 - enlarge(y2, eta)),
            lam3_km1 * (px_km1 - enlarge(x3, eta)),
            lam4_km1 * (py_km1 - enlarge(y1, eta)),
            lam5_km1 * (px_km1 - enlarge(x2, eta)),
            lam6_km1 * (py_km1 - enlarge(y1, eta)),
            lam7_km1 * (px_km1 - enlarge(x1, eta)),
            lam8_km1 * (py_km1 - enlarge(y2, eta))
        ]
        ineq_con_mode_2 = [
            lam1_km1 * (reduce_func(x4, eta) - px_km1),
            lam2_km1 * (reduce_func(y3, eta) - py_km1),
            lam3_km1 * (reduce_func(x4, eta) - px_km1),
            lam4_km1 * (reduce_func(y2, eta) - py_km1),
            lam5_km1 * (reduce_func(x3, eta) - px_km1),
            lam6_km1 * (reduce_func(y2, eta) - py_km1),
            lam7_km1 * (reduce_func(x2, eta) - px_km1),
            lam8_km1 * (reduce_func(y3, eta) - py_km1)
        ]
        ineq_con_mode_3 = [
            lam1_km1 * (-Fy_km1),
            lam2_km1 * (-Fx_km1),
            lam3_km1 * (Fy_km1),
            lam4_km1 * (-Fx_km1),
            lam5_km1 * (Fy_km1),
            lam6_km1 * (Fx_km1),
            lam7_km1 * (Fy_km1),
            lam8_km1 * (Fx_km1)
        ]
        ineq = [ineq1] + ineq_con_mode + ineq_con_mode_2 + ineq_con_mode_3
    return ineq


def vertex_loss(vertices, rc, rs, sx, sy):
    """
    Computes a "vertex loss" by summing over all vertices.
    `vertices` is assumed to be a NumPy array with shape (n_vertices, 2).
    """
    total_loss = 0
    for i in range(vertices.shape[0]):
        vx = vertices[i, 0]
        vy = vertices[i, 1]
        l1 = (sx + vx * rc - vy * rs - vx)**2
        l2 = (sy + vx * rs + vy * rc - vy)**2
        total_loss += l1 + l2
    return total_loss


####################
# Main Application #
####################

def main():
    total_start = time.time()

    # --- CSTSS parameters ---
    if_mex = True
    params = {}
    params['if_mex'] = if_mex
    kappa = 2
    params['kappa'] = kappa
    relax_mode = "SOS"
    params['relax_mode'] = relax_mode
    cs_mode = "MD"
    params['cs_mode'] = cs_mode
    ts_mode = "NON"
    params['ts_mode'] = ts_mode
    ts_mom_mode = "NON"
    params['ts_mom_mode'] = ts_mom_mode
    ts_eq_mode = "NON"
    params['ts_eq_mode'] = ts_eq_mode
    if_solve = True
    params['if_solve'] = if_solve

    # --- Hyper-parameters ---
    if_smart_loss = False
    params['if_smart_loss'] = if_smart_loss
    if_sum_ineq = True
    params['if_sum_ineq'] = if_sum_ineq
    if_sum_eq = if_sum_ineq
    params['if_sum_eq'] = if_sum_eq

    # --- System parameters ---
    N = 30
    params['N'] = N
    dt = 0.1
    params['dt'] = dt
    m = 1.0
    params['m'] = m
    g = 9.8
    params['g'] = g
    mu1 = 0.3
    params['mu1'] = mu1
    mu2 = 0.2
    params['mu2'] = mu2
    c = 0.6
    params['c'] = c

    # --- Geometry information ---
    l = 0.05
    params['l'] = l
    dc = 37/14
    params['dc'] = dc
    r_val = l * max(np.sqrt(dc**2 + 0.5**2), np.sqrt((4 - dc)**2 + 2**2))
    params['r'] = r_val
    eta = 0.01
    params['eta'] = eta

    # --- Variable maximums and minimums ---
    s_max = 0.6
    params['s_max'] = s_max
    px_max = 2 * l
    params['px_max'] = px_max
    py_max = max(4 - dc, dc) * l
    params['py_max'] = py_max
    F_max = 1.0
    params['F_max'] = F_max
    fc_min = 0.7
    params['fc_min'] = fc_min

    # --- Initial states ---
    th_init = np.pi/3
    params['th_init'] = th_init
    sx_init = 0.2
    params['sx_init'] = sx_init
    sy_init = 0.1
    params['sy_init'] = sy_init

    # --- Vertices ---
    x1 = -2.0 * l; params['x1'] = x1
    x2 = -0.5 * l; params['x2'] = x2
    x3 = 0.5 * l;  params['x3'] = x3
    x4 = 2.0 * l;  params['x4'] = x4
    y1 = -dc * l; params['y1'] = y1
    y2 = (3.0 - dc) * l; params['y2'] = y2
    y3 = (4.0 - dc) * l; params['y3'] = y3
    vertices = np.array([
        [x4, y3], [x4, y2], [x3, y2], [x3, y1],
        [x2, y1], [x2, y2], [x1, y2], [x1, y3]
    ])
    params['vertices'] = vertices

    # --- Final states ---
    th_final = 0.0
    params['th_final'] = th_final
    sx_final = 0.0
    params['sx_final'] = sx_final
    sy_final = 0.0
    params['sy_final'] = sy_final

    # --- Objective coefficients ---
    th_coeff = 1.0; params['th_coeff'] = th_coeff
    s_coeff = 1.0; params['s_coeff'] = s_coeff
    F_coeff = 0.5; params['F_coeff'] = F_coeff
    vertex_coeff = 1.0; params['vertex_coeff'] = vertex_coeff
    continuous_coeff = 0.2; params['continuous_coeff'] = continuous_coeff
    final_penalty = 10.0; params['final_penalty'] = final_penalty

    # --- Total variable number ---
    total_var_num = 18 * N + 4
    params['total_var_num'] = total_var_num
    params['id'] = lambda prefix, k: get_id(prefix, k, N)
    var_mapping = get_var_mapping(params)
    params['var_mapping'] = var_mapping

    # --- Get remapping information ---
    ids_remap = get_remapped_ids(params)
    params['ids_remap'] = ids_remap

    # --- File management ---
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix_str = "PushT_Python/" + current_time + "/"
    for directory in ["./data/" + prefix_str, "./markdown/" + prefix_str,
                      "./figs/" + prefix_str, "./logs/" + prefix_str]:
        os.makedirs(directory, exist_ok=True)

    log_path = "./logs/" + prefix_str + "log.txt"
    with open(log_path, "w") as log_file:
        log_file.write("params: \n")
        log_file.write(str(params) + "\n")

    # --- Create NumPolySystem ---
    ps = NumPolySystem(n_vars=total_var_num)

    def v(prefix, k):
        return ps.var(params['id'](prefix, k) - 1)

    eq_mask_sys = []

    # Initial state constraints
    eqs, eq_mask = get_eq_init(
        v("rc", 0), v("rs", 0), v("sx", 0) * s_max, v("sy", 0) * s_max,
        th_init, sx_init, sy_init,
        dt, m, g, mu1, mu2, c, l, dc, r_val, fc_min, eta
    )
    for eq in eqs:
        ps.add_eq(eq)
    eq_mask_sys.extend(eq_mask)

    # Dynamics equality constraints
    for k in range(1, N + 1):
        eqs, eq_mask = get_eq_dyn(
            v("rc", k - 1), v("rc", k),
            v("rs", k - 1), v("rs", k),
            v("sx", k - 1) * s_max, v("sx", k) * s_max,
            v("sy", k - 1) * s_max, v("sy", k) * s_max,
            v("fc", k), v("fs", k),
            v("px", k) * px_max, v("py", k) * py_max,
            v("Fx", k) * F_max, v("Fy", k) * F_max,
            v("lam1", k), v("lam2", k),
            v("lam3", k), v("lam4", k),
            v("lam5", k), v("lam6", k),
            v("lam7", k), v("lam8", k),
            dt, m, g, mu1, mu2, c, l, dc, r_val, fc_min, eta, if_sum_eq
        )
        for eq in eqs:
            ps.add_eq(eq)
        eq_mask_sys.extend(eq_mask)

    # Dynamics inequality constraints
    for k in range(1, N + 1):
        ineqs = get_ineq_dyn(
            v("rc", k - 1), v("rc", k),
            v("rs", k - 1), v("rs", k),
            v("sx", k - 1) * s_max, v("sx", k) * s_max,
            v("sy", k - 1) * s_max, v("sy", k) * s_max,
            v("fc", k), v("fs", k),
            v("px", k) * px_max, v("py", k) * py_max,
            v("Fx", k) * F_max, v("Fy", k) * F_max,
            v("lam1", k), v("lam2", k),
            v("lam3", k), v("lam4", k),
            v("lam5", k), v("lam6", k),
            v("lam7", k), v("lam8", k),
            dt, m, g, mu1, mu2, c, l, dc, r_val, fc_min, eta, if_sum_ineq
        )
        for ineq in ineqs:
            ps.add_ineq(ineq)

        if k == 1:
            ps.add_ineq(1 - v("sx", k - 1)**2)
            ps.add_ineq(1 - v("sy", k - 1)**2)
        ps.add_ineq(1 - v("sx", k)**2)
        ps.add_ineq(1 - v("sy", k)**2)
        ps.add_ineq(1 - v("px", k)**2)
        ps.add_ineq(1 - v("py", k)**2)
        ps.add_ineq(1 - v("Fx", k)**2)
        ps.add_ineq(1 - v("Fy", k)**2)

    # Objective
    sx_final_scaled = sx_final / s_max
    sy_final_scaled = sy_final / s_max
    rc_final = np.cos(th_final)
    rs_final = np.sin(th_final)

    obj_expr = 0
    for k in range(1, N + 1):
        if k < N:
            if if_smart_loss:
                loss_expr = (vertex_coeff *
                             vertex_loss(vertices,
                                         v("rc", k), v("rs", k),
                                         v("sx", k) * s_max, v("sy", k) * s_max))
                loss_expr = loss_expr + F_coeff * (v("Fx", k)**2 + v("Fy", k)**2)
                loss_expr = loss_expr + continuous_coeff * (v("px", k + 1) - v("px", k))**2
                loss_expr = loss_expr + continuous_coeff * (v("py", k + 1) - v("py", k))**2
                obj_expr = obj_expr + loss_expr
            else:
                expr = (th_coeff * (v("rc", k) - rc_final)**2 +
                        th_coeff * (v("rs", k) - rs_final)**2 +
                        s_coeff * (v("sx", k) - sx_final_scaled)**2 +
                        s_coeff * (v("sy", k) - sy_final_scaled)**2 +
                        continuous_coeff * (v("px", k + 1) - v("px", k))**2 +
                        continuous_coeff * (v("py", k + 1) - v("py", k))**2 +
                        F_coeff * (v("Fx", k)**2 + v("Fy", k)**2))
                obj_expr = obj_expr + expr
        else:
            if if_smart_loss:
                expr = (final_penalty * vertex_coeff *
                        vertex_loss(vertices,
                                    v("rc", k), v("rs", k),
                                    v("sx", k) * s_max, v("sy", k) * s_max) +
                        F_coeff * (v("Fx", k)**2 + v("Fy", k)**2))
                obj_expr = obj_expr + expr
            else:
                expr = (final_penalty * ((v("rc", k) - rc_final)**2 +
                                         (v("rs", k) - rs_final)**2 +
                                         (v("sx", k) - sx_final_scaled)**2 +
                                         (v("sy", k) - sy_final_scaled)**2) +
                        F_coeff * (v("Fx", k)**2 + v("Fy", k)**2))
                obj_expr = obj_expr + expr

    ps.set_obj(obj_expr)

    # --- Clean polynomials ---
    ps.clean_all(tol=1e-14, if_scale=True, scale_obj = False)

    # --- Get supp_rpt data ---
    poly_data = ps.get_supp_rpt_data(kappa)

    # --- Initialize cliques ---
    cliques = []
    for k in range(1, N + 1):
        if k > 1 and abs(continuous_coeff) > 1e-6:
            clique = [params['id']("rc", k - 1), params['id']("rc", k),
                      params['id']("rs", k - 1), params['id']("rs", k),
                      params['id']("sx", k - 1), params['id']("sx", k),
                      params['id']("sy", k - 1), params['id']("sy", k),
                      params['id']("fc", k), params['id']("fs", k),
                      params['id']("px", k - 1), params['id']("px", k),
                      params['id']("py", k - 1), params['id']("py", k),
                      params['id']("Fx", k), params['id']("Fy", k),
                      params['id']("lam1", k), params['id']("lam2", k),
                      params['id']("lam3", k), params['id']("lam4", k),
                      params['id']("lam5", k), params['id']("lam6", k),
                      params['id']("lam7", k), params['id']("lam8", k)]
        else:
            clique = [params['id']("rc", k - 1), params['id']("rc", k),
                      params['id']("rs", k - 1), params['id']("rs", k),
                      params['id']("sx", k - 1), params['id']("sx", k),
                      params['id']("sy", k - 1), params['id']("sy", k),
                      params['id']("fc", k), params['id']("fs", k),
                      params['id']("px", k), params['id']("py", k),
                      params['id']("Fx", k), params['id']("Fy", k),
                      params['id']("lam1", k), params['id']("lam2", k),
                      params['id']("lam3", k), params['id']("lam4", k),
                      params['id']("lam5", k), params['id']("lam6", k),
                      params['id']("lam7", k), params['id']("lam8", k)]
        cliques.append(clique)
    params['cliques'] = cliques

    print("Construction Finish!")

    # --- Run CSTSS ---
    start_time = time.time()
    result, res, coeff_info, aux_info = CSTSS_pybind(
        poly_data, kappa, total_var_num, params
    )
    elapsed_time = time.time() - start_time
    aux_info["result"] = result
    params["aux_info"] = aux_info

    # Log result and timings.
    with open(log_path, "a") as log_file:
        log_file.write(f"PushT  N={N}, Relax={relax_mode}, TS={ts_mode}, CS={cs_mode}, "
                       f"result={result:.20f}, operation time={elapsed_time:.5f}, "
                       f"mosek time={aux_info.get('mosek_time', 0):.5f},\n")

    # --- Remap clique ids ---
    if "cliques" in aux_info and aux_info["cliques"]:
        cliques_remapped = []
        aver_remapped = []
        for clique in aux_info["cliques"]:
            remapped = [params["ids_remap"][i - 1] for i in clique]
            cliques_remapped.append(remapped)
            aver_remapped.append(np.mean(remapped))
        cliques_rank = np.argsort(aver_remapped)
        params["cliques_rank"] = cliques_rank
    else:
        params["cliques_rank"] = []

    # --- Markdown debug ---
    # Build clique visualization: each clique becomes a polynomial summing its variables
    clique_supp_list = []
    clique_coeff_list = []
    kappa_width = 2 * kappa
    if "cliques" in aux_info and aux_info["cliques"]:
        cliques_rank = params["cliques_rank"]
        for i in range(len(cliques_rank)):
            ii = cliques_rank[i]
            sorted_vars = sorted(aux_info["cliques"][ii])
            # Each variable j becomes a degree-1 term: index sequence [0,...,0, j]
            supp = np.zeros((len(sorted_vars), kappa_width), dtype=np.float64)
            for idx_v, j in enumerate(sorted_vars):
                supp[idx_v, -1] = j  # 1-indexed, right-aligned
            clique_supp_list.append(supp)
            clique_coeff_list.append(np.ones(len(sorted_vars)))

    md_path = "./markdown/" + prefix_str + "opt_problem.md"
    with open(md_path, "w") as md_file:
        md_file.write("equality constraints: \n")
        numpoly_visualize(aux_info['supp_rpt_h'], aux_info['coeff_h'], var_mapping, md_file)
        md_file.write("inequality constraints: \n")
        numpoly_visualize(aux_info['supp_rpt_g'], aux_info['coeff_g'], var_mapping, md_file)
        md_file.write("objective: \n")
        numpoly_visualize([aux_info['supp_rpt_f']], [aux_info['coeff_f']], var_mapping, md_file)
        md_file.write("cliques: \n")
        numpoly_visualize(clique_supp_list, clique_coeff_list, var_mapping, md_file)

    if not params['if_solve']:
        params['self_cliques'] = cliques
        with open("./data/" + prefix_str + "params.pkl", "wb") as f:
            pickle.dump(params, f)
        return

    # --- Extract solution ---
    Xopt = res['Xopt']
    yopt = res['yopt']
    Sopt = res['Sopt']

    if relax_mode == 'MOMENT':
        Xs = Xopt
    elif relax_mode == 'SOS':
        Xs = []
        for S in Sopt:
            Xs.append(-S)

    ts_info = aux_info["ts_info"]
    cliques = aux_info["cliques"]
    mon_rpt = aux_info["mon_rpt"]

    mom_mat_num = sum(len(ts_info[i]) for i in range(len(cliques)))
    mom_mat_rpt = [None] * mom_mat_num

    idx = 0
    for i in range(len(cliques)):
        for j in range(len(ts_info[i])):
            rpt = mon_rpt[i][ts_info[i][j], :]
            rpt = np.hstack([np.zeros_like(rpt), rpt])
            mom_mat_rpt[idx] = rpt
            idx += 1

    if ts_mode == "NON":
        v_opt_robust, output_info_robust = robust_extract_CS(Xs, mom_mat_rpt, total_var_num, 1e-2)
        with open("./data/" + prefix_str + "v_opt_robust.pkl", "wb") as f:
            pickle.dump(v_opt_robust, f)
        v_opt_ordered, output_info_ordered = ordered_extract_CS(Xs, mom_mat_rpt, total_var_num, 1e-2, params.get("cliques_rank", []))
        with open("./data/" + prefix_str + "v_opt_ordered.pkl", "wb") as f:
            pickle.dump(v_opt_ordered, f)
        print(v_opt_ordered)

    v_opt_naive, output_info_naive = naive_extract(Xs, aux_info['mon_rpt'], aux_info['ts_info'], total_var_num)
    print(v_opt_naive)
    with open("./data/" + prefix_str + "v_opt_naive.pkl", "wb") as f:
        pickle.dump(v_opt_naive, f)

    with open("./data/" + prefix_str + "data.pkl", "wb") as f:
        pickle.dump({'aux_info': aux_info, 'mom_mat_rpt': mom_mat_rpt,
                     'mom_mat_num': mom_mat_num, 'total_var_num': total_var_num}, f)

    supp_rpt_f = aux_info.get("supp_rpt_f", None)
    supp_rpt_g = aux_info.get("supp_rpt_g", None)
    supp_rpt_h = aux_info.get("supp_rpt_h", None)
    coeff_f = aux_info.get("coeff_f", None)
    coeff_g = aux_info.get("coeff_g", None)
    coeff_h = aux_info.get("coeff_h", None)
    with open("./data/" + prefix_str + "polys.pkl", "wb") as f:
        pickle.dump({'supp_rpt_f': supp_rpt_f, 'supp_rpt_g': supp_rpt_g,
                     'supp_rpt_h': supp_rpt_h, 'coeff_f': coeff_f,
                     'coeff_g': coeff_g, 'coeff_h': coeff_h}, f)
    params['self_cliques'] = cliques

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.5f} s")
    with open(log_path, "a") as log_file:
        log_file.write(f"total time={total_time:.5f}\n")


if __name__ == "__main__":
    main()
