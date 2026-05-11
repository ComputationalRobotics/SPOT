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

# Long groups (N+1 entries): x, y, rc, rs, rcld, rsld, rclu, rslu, rcrd, rsrd, rcru, rsru
# Short groups (N entries): fc, fs, fcld, fsld, fclu, fslu, fcrd, fsrd, fcru, fsru,
#                            xl, yl, vxl, vyl, xr, yr, vxr, vyr,
#                            dl, vrell, lamnl, lamtl, dr, vrelr, lamnr, lamtr

LONG_PREFIXES = ["x", "y", "rc", "rs", "rcld", "rsld", "rclu", "rslu", "rcrd", "rsrd", "rcru", "rsru"]
SHORT_PREFIXES = ["fc", "fs", "fcld", "fsld", "fclu", "fslu", "fcrd", "fsrd", "fcru", "fsru",
                  "xl", "yl", "vxl", "vyl", "xr", "yr", "vxr", "vyr",
                  "dl", "vrell", "lamnl", "lamtl", "dr", "vrelr", "lamnr", "lamtr"]


def get_var_mapping_and_dict(N):
    var_start_dict = {}
    var_mapping = {}
    cnt = 1
    long_list = list(range(N + 1))
    short_list = list(range(N))

    # Formatting map for var_mapping
    fmt = {
        "x": "x_{{{k}}}", "y": "y_{{{k}}}",
        "rc": "r_{{c, {k}}}", "rs": "r_{{s, {k}}}",
        "fc": "f_{{c, {k}}}", "fs": "f_{{s, {k}}}",
        "rcld": "r_{{c, ld, {k}}}", "rsld": "r_{{s, ld, {k}}}",
        "fcld": "f_{{c, ld, {k}}}", "fsld": "f_{{s, ld, {k}}}",
        "rclu": "r_{{c, lu, {k}}}", "rslu": "r_{{s, lu, {k}}}",
        "fclu": "f_{{c, lu, {k}}}", "fslu": "f_{{s, lu, {k}}}",
        "xl": "x_{{l, {k}}}", "yl": "y_{{l, {k}}}",
        "vxl": "v_{{x, l, {k}}}", "vyl": "v_{{y, l, {k}}}",
        "rcrd": "r_{{c, rd, {k}}}", "rsrd": "r_{{s, rd, {k}}}",
        "fcrd": "f_{{c, rd, {k}}}", "fsrd": "f_{{s, rd, {k}}}",
        "rcru": "r_{{c, ru, {k}}}", "rsru": "r_{{s, ru, {k}}}",
        "fcru": "f_{{c, ru, {k}}}", "fsru": "f_{{s, ru, {k}}}",
        "xr": "x_{{r, {k}}}", "yr": "y_{{r, {k}}}",
        "vxr": "v_{{x, r, {k}}}", "vyr": "v_{{y, r, {k}}}",
        "dl": "d_{{l, {k}}}", "vrell": "v_{{rel, l, {k}}}",
        "lamnl": "\\lambda_{{n, l, {k}}}", "lamtl": "\\lambda_{{t, l, {k}}}",
        "dr": "d_{{r, {k}}}", "vrelr": "v_{{rel, r, {k}}}",
        "lamnr": "\\lambda_{{n, r, {k}}}", "lamtr": "\\lambda_{{t, r, {k}}}",
    }

    # Order matches MATLAB: circle long, circle short, left long, left short, left pos/vel,
    #                        right long, right short, right pos/vel, left contact, right contact
    ordered_prefixes = [
        # circle
        ("x", long_list), ("y", long_list), ("rc", long_list), ("rs", long_list),
        ("fc", short_list), ("fs", short_list),
        # left finger
        ("rcld", long_list), ("rsld", long_list), ("fcld", short_list), ("fsld", short_list),
        ("rclu", long_list), ("rslu", long_list), ("fclu", short_list), ("fslu", short_list),
        ("xl", short_list), ("yl", short_list), ("vxl", short_list), ("vyl", short_list),
        # right finger
        ("rcrd", long_list), ("rsrd", long_list), ("fcrd", short_list), ("fsrd", short_list),
        ("rcru", long_list), ("rsru", long_list), ("fcru", short_list), ("fsru", short_list),
        ("xr", short_list), ("yr", short_list), ("vxr", short_list), ("vyr", short_list),
        # left contact
        ("dl", short_list), ("vrell", short_list), ("lamnl", short_list), ("lamtl", short_list),
        # right contact
        ("dr", short_list), ("vrelr", short_list), ("lamnr", short_list), ("lamtr", short_list),
    ]

    for prefix, klist in ordered_prefixes:
        var_start_dict[prefix] = cnt
        for k in klist:
            var_mapping[cnt] = fmt[prefix].format(k=k)
            cnt += 1

    total_var_num = cnt - 1
    var_start_dict["N"] = N
    return var_mapping, var_start_dict, total_var_num


def get_id(prefix, k, var_start_dict):
    return var_start_dict[prefix] + k


def get_remapped_ids(params):
    N = params['N']
    total_var_num = params['total_var_num']
    var_start_dict = params['var_start_dict']
    id_func = params['id']
    ids_remap = np.zeros(total_var_num, dtype=int)
    idx = 1
    for k in range(N + 1):
        ids_remap[id_func("x", k) - 1] = idx; idx += 1
        ids_remap[id_func("y", k) - 1] = idx; idx += 1
        ids_remap[id_func("rc", k) - 1] = idx; idx += 1
        ids_remap[id_func("rs", k) - 1] = idx; idx += 1
        if k < N:
            ids_remap[id_func("fc", k) - 1] = idx; idx += 1
            ids_remap[id_func("fs", k) - 1] = idx; idx += 1
        ids_remap[id_func("rcld", k) - 1] = idx; idx += 1
        ids_remap[id_func("rsld", k) - 1] = idx; idx += 1
        ids_remap[id_func("rclu", k) - 1] = idx; idx += 1
        ids_remap[id_func("rslu", k) - 1] = idx; idx += 1
        ids_remap[id_func("rcrd", k) - 1] = idx; idx += 1
        ids_remap[id_func("rsrd", k) - 1] = idx; idx += 1
        ids_remap[id_func("rcru", k) - 1] = idx; idx += 1
        ids_remap[id_func("rsru", k) - 1] = idx; idx += 1
        if k < N:
            ids_remap[id_func("fcld", k) - 1] = idx; idx += 1
            ids_remap[id_func("fsld", k) - 1] = idx; idx += 1
            ids_remap[id_func("fclu", k) - 1] = idx; idx += 1
            ids_remap[id_func("fslu", k) - 1] = idx; idx += 1
            ids_remap[id_func("fcrd", k) - 1] = idx; idx += 1
            ids_remap[id_func("fsrd", k) - 1] = idx; idx += 1
            ids_remap[id_func("fcru", k) - 1] = idx; idx += 1
            ids_remap[id_func("fsru", k) - 1] = idx; idx += 1
            ids_remap[id_func("xl", k) - 1] = idx; idx += 1
            ids_remap[id_func("yl", k) - 1] = idx; idx += 1
            ids_remap[id_func("vxl", k) - 1] = idx; idx += 1
            ids_remap[id_func("vyl", k) - 1] = idx; idx += 1
            ids_remap[id_func("xr", k) - 1] = idx; idx += 1
            ids_remap[id_func("yr", k) - 1] = idx; idx += 1
            ids_remap[id_func("vxr", k) - 1] = idx; idx += 1
            ids_remap[id_func("vyr", k) - 1] = idx; idx += 1
            ids_remap[id_func("dl", k) - 1] = idx; idx += 1
            ids_remap[id_func("vrell", k) - 1] = idx; idx += 1
            ids_remap[id_func("lamnl", k) - 1] = idx; idx += 1
            ids_remap[id_func("lamtl", k) - 1] = idx; idx += 1
            ids_remap[id_func("dr", k) - 1] = idx; idx += 1
            ids_remap[id_func("vrelr", k) - 1] = idx; idx += 1
            ids_remap[id_func("lamnr", k) - 1] = idx; idx += 1
            ids_remap[id_func("lamtr", k) - 1] = idx; idx += 1
    return ids_remap


# ---- 8 constraint functions ----

def get_init_constraints(x_0, y_0, rc_0, rs_0,
                         rcld_0, rsld_0, rclu_0, rslu_0,
                         rcrd_0, rsrd_0, rcru_0, rsru_0,
                         params):
    eqs = [
        x_0 - params['x_init'],
        y_0 - params['y_init'],
        rc_0 - np.cos(params['th_init']),
        rs_0 - np.sin(params['th_init']),
        rcld_0 - np.cos(params['th_ld_init']),
        rsld_0 - np.sin(params['th_ld_init']),
        rclu_0 - np.cos(params['th_lu_init']),
        rslu_0 - np.sin(params['th_lu_init']),
        rcrd_0 - np.cos(params['th_rd_init']),
        rsrd_0 - np.sin(params['th_rd_init']),
        rcru_0 - np.cos(params['th_ru_init']),
        rsru_0 - np.sin(params['th_ru_init']),
        rc_0**2 + rs_0**2 - 1,
        rcld_0**2 + rsld_0**2 - 1,
        rclu_0**2 + rslu_0**2 - 1,
        rcrd_0**2 + rsrd_0**2 - 1,
        rcru_0**2 + rsru_0**2 - 1,
    ]
    eq_mask = [1]*12 + [0]*5
    ineqs = []
    return eqs, ineqs, eq_mask


def get_right_finger_kinematics(rcrd_km1, rcrd_k, rsrd_km1, rsrd_k, fcrd_km1, fsrd_km1,
                                 rcru_km1, rcru_k, rsru_km1, rsru_k, fcru_km1, fsru_km1,
                                 xr_km1, yr_km1, vxr_km1, vyr_km1,
                                 params):
    H = params['H']; Ld = params['Ld']; Lu = params['Lu']; dt = params['dt']
    eqs = [
        Ld * rcrd_km1 + Lu * rcru_km1 + H / 2 - xr_km1,
        Ld * rsrd_km1 + Lu * rsru_km1 - yr_km1,
        -Ld / dt * rsrd_km1 * fsrd_km1 - Lu / dt * rsru_km1 * fsru_km1 - vxr_km1,
        Ld / dt * rcrd_km1 * fsrd_km1 + Lu / dt * rcru_km1 * fsru_km1 - vyr_km1,
        rcrd_km1 * fcrd_km1 - rsrd_km1 * fsrd_km1 - rcrd_k,
        rcrd_km1 * fsrd_km1 + rsrd_km1 * fcrd_km1 - rsrd_k,
        rcru_km1 * fcru_km1 - rsru_km1 * fsru_km1 - rcru_k,
        rcru_km1 * fsru_km1 + rsru_km1 * fcru_km1 - rsru_k,
        rcrd_k**2 + rsrd_k**2 - 1,
        rcru_k**2 + rsru_k**2 - 1,
        fcrd_km1**2 + fsrd_km1**2 - 1,
        fcru_km1**2 + fsru_km1**2 - 1,
    ]
    eq_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
    ineqs = [
        fcrd_km1 - params['fcrd_min'],
        fcru_km1 - params['fcru_min'],
    ]
    return eqs, ineqs, eq_mask


def get_left_finger_kinematics(rcld_km1, rcld_k, rsld_km1, rsld_k, fcld_km1, fsld_km1,
                                rclu_km1, rclu_k, rslu_km1, rslu_k, fclu_km1, fslu_km1,
                                xl_km1, yl_km1, vxl_km1, vyl_km1,
                                params):
    H = params['H']; Ld = params['Ld']; Lu = params['Lu']; dt = params['dt']
    eqs = [
        Ld * rcld_km1 + Lu * rclu_km1 - H / 2 - xl_km1,
        Ld * rsld_km1 + Lu * rslu_km1 - yl_km1,
        -Ld / dt * rsld_km1 * fsld_km1 - Lu / dt * rslu_km1 * fslu_km1 - vxl_km1,
        Ld / dt * rcld_km1 * fsld_km1 + Lu / dt * rclu_km1 * fslu_km1 - vyl_km1,
        rcld_km1 * fcld_km1 - rsld_km1 * fsld_km1 - rcld_k,
        rcld_km1 * fsld_km1 + rsld_km1 * fcld_km1 - rsld_k,
        rclu_km1 * fclu_km1 - rslu_km1 * fslu_km1 - rclu_k,
        rclu_km1 * fslu_km1 + rslu_km1 * fclu_km1 - rslu_k,
        rcld_k**2 + rsld_k**2 - 1,
        rclu_k**2 + rslu_k**2 - 1,
        fcld_km1**2 + fsld_km1**2 - 1,
        fclu_km1**2 + fslu_km1**2 - 1,
    ]
    eq_mask = [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1]
    ineqs = [
        fcld_km1 - params['fcld_min'],
        fclu_km1 - params['fclu_min'],
    ]
    return eqs, ineqs, eq_mask


def get_self_collision_info(rcld_k, rsld_k, rclu_k, rslu_k,
                             rcrd_k, rsrd_k, rcru_k, rsru_k,
                             params):
    th0 = params['theta_0']; r = params['r']; H = params['H']
    Ld = params['Ld']; Lu = params['Lu']
    eqs = []
    eq_mask = []
    ineqs = [
        rsld_k - np.sin(th0),
        rsrd_k - np.sin(th0),
        rsld_k * rclu_k - rcld_k * rslu_k,
        rcld_k * rclu_k + rsld_k * rslu_k + np.cos(2 * th0),
        rsru_k * rcrd_k - rcru_k * rsrd_k,
        rcru_k * rcrd_k + rsru_k * rsrd_k + np.cos(2 * th0),
        -(Ld * rcld_k + Lu * rclu_k - H / 2) - r,
        (Ld * rcrd_k + Lu * rcru_k + H / 2) - r,
    ]
    return eqs, ineqs, eq_mask


def get_right_finger_contact(x_km1, x_k, y_km1, y_k, fs_km1,
                              xr_km1, yr_km1, vxr_km1, vyr_km1,
                              dr_km1, vrelr_km1, lamnr_km1, lamtr_km1,
                              params):
    r = params['r']; R = params['R']; dt = params['dt']; mu = params['mu']
    vx_km1 = (x_k - x_km1) / dt
    vy_km1 = (y_k - y_km1) / dt
    cos_etar = (xr_km1 - x_km1) / (R + r)
    sin_etar = (yr_km1 - y_km1) / (R + r)
    eqs = [
        (xr_km1 - x_km1)**2 + (yr_km1 - y_km1)**2 - dr_km1**2,
        -(vxr_km1 - vx_km1) * sin_etar + (vyr_km1 - vy_km1) * cos_etar - R / dt * fs_km1 - vrelr_km1,
        (dr_km1 - R - r) * lamnr_km1,
        vrelr_km1 * (mu**2 * lamnr_km1**2 - lamtr_km1**2),
    ]
    eq_mask = [1, 1, 1, 1]
    ineqs = [
        dr_km1 - (R + r),
        -lamnr_km1,
        mu**2 * lamnr_km1**2 - lamtr_km1**2,
        vrelr_km1 * lamtr_km1,
    ]
    return eqs, ineqs, eq_mask


def get_left_finger_contact(x_km1, x_k, y_km1, y_k, fs_km1,
                             xl_km1, yl_km1, vxl_km1, vyl_km1,
                             dl_km1, vrell_km1, lamnl_km1, lamtl_km1,
                             params):
    r = params['r']; R = params['R']; dt = params['dt']; mu = params['mu']
    vx_km1 = (x_k - x_km1) / dt
    vy_km1 = (y_k - y_km1) / dt
    cos_etal = (xl_km1 - x_km1) / (R + r)
    sin_etal = (yl_km1 - y_km1) / (R + r)
    eqs = [
        (xl_km1 - x_km1)**2 + (yl_km1 - y_km1)**2 - dl_km1**2,
        -(vxl_km1 - vx_km1) * sin_etal + (vyl_km1 - vy_km1) * cos_etal - R / dt * fs_km1 - vrell_km1,
        (dl_km1 - R - r) * lamnl_km1,
        vrell_km1 * (mu**2 * lamnl_km1**2 - lamtl_km1**2),
    ]
    eq_mask = [1, 1, 1, 1]
    ineqs = [
        dl_km1 - (R + r),
        -lamnl_km1,
        mu**2 * lamnl_km1**2 - lamtl_km1**2,
        vrell_km1 * lamtl_km1,
    ]
    return eqs, ineqs, eq_mask


def get_circle_dynamics(x_km1, x_k, y_km1, y_k,
                         rc_km1, rc_k, rs_km1, rs_k, fc_km1, fs_km1,
                         xl_km1, yl_km1, xr_km1, yr_km1,
                         lamnl_km1, lamtl_km1, lamnr_km1, lamtr_km1,
                         params):
    r = params['r']; R = params['R']; dt = params['dt']; c = params['c']
    vx_km1 = (x_k - x_km1) / dt
    vy_km1 = (y_k - y_km1) / dt
    cos_etal = (xl_km1 - x_km1) / (R + r)
    sin_etal = (yl_km1 - y_km1) / (R + r)
    cos_etar = (xr_km1 - x_km1) / (R + r)
    sin_etar = (yr_km1 - y_km1) / (R + r)
    eqs = [
        lamnr_km1 * cos_etar - lamtr_km1 * sin_etar + lamnl_km1 * cos_etal - lamtl_km1 * sin_etal - vx_km1,
        lamnr_km1 * sin_etar + lamtr_km1 * cos_etar + lamnl_km1 * sin_etal + lamtl_km1 * cos_etal - vy_km1,
        dt / (c * R) * (lamtr_km1 + lamtl_km1) - fs_km1,
        rc_km1 * fc_km1 - rs_km1 * fs_km1 - rc_k,
        rc_km1 * fs_km1 + rs_km1 * fc_km1 - rs_k,
        rc_k**2 + rs_k**2 - 1,
        fc_km1**2 + fs_km1**2 - 1,
    ]
    eq_mask = [1, 1, 1, 1, 1, 1, 1]
    ineqs = [
        fc_km1 - params['fc_min'],
    ]
    return eqs, ineqs, eq_mask


def get_collision_avoidance_info(x_k, y_k,
                                  rcld_k, rsld_k, rclu_k, rslu_k,
                                  rcrd_k, rsrd_k, rcru_k, rsru_k,
                                  params):
    r = params['r']; R = params['R']; l = params['l']
    Ld = params['Ld']; Lu = params['Lu']; H = params['H']
    eqs = []
    eq_mask = []
    ineqs = [
        ((l + r) * rcrd_k + H / 2 - x_k)**2 + ((l + r) * rsrd_k - y_k)**2 - (R + r)**2,
        ((2 * l + 3 * r) * rcrd_k + H / 2 - x_k)**2 + ((2 * l + 3 * r) * rsrd_k - y_k)**2 - (R + r)**2,
        (Ld * rcrd_k + (l + r) * rcru_k + H / 2 - x_k)**2 + (Ld * rsrd_k + (l + r) * rsru_k - y_k)**2 - (R + r)**2,
        (Ld * rcrd_k + (2 * l + 3 * r) * rcru_k + H / 2 - x_k)**2 + (Ld * rsrd_k + (2 * l + 3 * r) * rsru_k - y_k)**2 - (R + r)**2,
        ((l + r) * rcld_k - H / 2 - x_k)**2 + ((l + r) * rsld_k - y_k)**2 - (R + r)**2,
        ((2 * l + 3 * r) * rcld_k - H / 2 - x_k)**2 + ((2 * l + 3 * r) * rsld_k - y_k)**2 - (R + r)**2,
        (Ld * rcld_k + (l + r) * rclu_k - H / 2 - x_k)**2 + (Ld * rsld_k + (l + r) * rslu_k - y_k)**2 - (R + r)**2,
        (Ld * rcld_k + (2 * l + 3 * r) * rclu_k - H / 2 - x_k)**2 + (Ld * rsld_k + (2 * l + 3 * r) * rslu_k - y_k)**2 - (R + r)**2,
    ]
    return eqs, ineqs, eq_mask


####################
# Main Application #
####################

def main():
    total_start = time.time()

    # --- CSTSS parameters ---
    params = {}
    kappa = 2; params['kappa'] = kappa
    relax_mode = "SOS"; params['relax_mode'] = relax_mode
    cs_mode = "SELF"; params['cs_mode'] = cs_mode
    ts_mode = "NON"; params['ts_mode'] = ts_mode
    ts_mom_mode = "NON"; params['ts_mom_mode'] = ts_mom_mode
    ts_eq_mode = "NON"; params['ts_eq_mode'] = ts_eq_mode
    if_solve = True; params['if_solve'] = if_solve
    if_mex = True; params['if_mex'] = if_mex

    # --- System parameters ---
    N = 30; params['N'] = N
    dt = 0.05; params['dt'] = dt

    # Variable mapping and indexing
    var_mapping, var_start_dict, total_var_num = get_var_mapping_and_dict(N)
    params['total_var_num'] = total_var_num
    params['var_mapping'] = var_mapping
    params['var_start_dict'] = var_start_dict
    params['id'] = lambda prefix, k: get_id(prefix, k, var_start_dict)

    # Geometric information
    l = 0.02; params['l'] = l
    r = 0.01; params['r'] = r
    theta_0 = np.arcsin(r / (r + l)); params['theta_0'] = theta_0
    Ld = 3 * l + 4 * r; params['Ld'] = Ld
    Lu = 2 * l + 3 * r; params['Lu'] = Lu
    H = 0.2; params['H'] = H
    R = 0.08; params['R'] = R

    # Dynamics
    mu = 0.3; params['mu'] = mu
    c = 0.6; params['c'] = c

    # Limits
    fc_min = 0.9; params['fc_min'] = fc_min
    fcrd_min = 0.9; params['fcrd_min'] = fcrd_min
    fcru_min = fcrd_min; params['fcru_min'] = fcru_min
    fcld_min = fcrd_min; params['fcld_min'] = fcld_min
    fclu_min = fcrd_min; params['fclu_min'] = fclu_min
    x_max = Ld + Lu + H / 2; params['x_max'] = x_max
    y_max = x_max; params['y_max'] = y_max
    xr_max = x_max; params['xr_max'] = xr_max
    yr_max = y_max; params['yr_max'] = yr_max
    xl_max = x_max; params['xl_max'] = xl_max
    yl_max = y_max; params['yl_max'] = yl_max
    vxr_max = 1.0; params['vxr_max'] = vxr_max
    vyr_max = vxr_max; params['vyr_max'] = vyr_max
    vxl_max = vxr_max; params['vxl_max'] = vxl_max
    vyl_max = vxr_max; params['vyl_max'] = vyl_max
    vx_max = vxr_max; params['vx_max'] = vx_max
    vy_max = vxr_max; params['vy_max'] = vy_max
    lamnr_max = 1.0; params['lamnr_max'] = lamnr_max
    lamtr_max = lamnr_max; params['lamtr_max'] = lamtr_max
    lamnl_max = lamnr_max; params['lamnl_max'] = lamnl_max
    lamtl_max = lamnr_max; params['lamtl_max'] = lamtl_max
    dr_max = x_max * 2; params['dr_max'] = dr_max
    dl_max = dr_max; params['dl_max'] = dl_max
    vrelr_max = 3.0; params['vrelr_max'] = vrelr_max
    vrell_max = vrelr_max; params['vrell_max'] = vrell_max

    # Initial states
    x_init = 0.0; params['x_init'] = x_init
    y_init = Ld; params['y_init'] = y_init
    th_init = 0.0; params['th_init'] = th_init
    th_ld_init = 2 * np.pi / 3; params['th_ld_init'] = th_ld_init
    th_lu_init = np.pi / 3; params['th_lu_init'] = th_lu_init
    th_rd_init = np.pi / 3; params['th_rd_init'] = th_rd_init
    th_ru_init = 2 * np.pi / 3; params['th_ru_init'] = th_ru_init

    # Loss coefficients
    loss_track_coeff = 1.0; params['loss_track_coeff'] = loss_track_coeff
    loss_track_th_list = np.linspace(0.0, 2 * np.pi, N); params['loss_track_th_list'] = loss_track_th_list
    loss_vel_finger_coeff = 1.0; params['loss_vel_finger_coeff'] = loss_vel_finger_coeff
    loss_vel_circle_coeff = 1.0; params['loss_vel_circle_coeff'] = loss_vel_circle_coeff
    loss_vel_relative_coeff = 0.0; params['loss_vel_relative_coeff'] = loss_vel_relative_coeff
    loss_force_coeff = 1.0; params['loss_force_coeff'] = loss_force_coeff
    loss_pos_finger_coeff = 0.1; params['loss_pos_finger_coeff'] = loss_pos_finger_coeff

    # --- Get remapping information ---
    ids_remap = get_remapped_ids(params); params['ids_remap'] = ids_remap

    # --- File management ---
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix_str = "PlanarHand_Python_Optimized/" + current_time + "/"
    for directory in ["./data/" + prefix_str, "./markdown/" + prefix_str,
                      "./figs/" + prefix_str, "./logs/" + prefix_str]:
        os.makedirs(directory, exist_ok=True)

    log_path = "./logs/" + prefix_str + "log.txt"
    with open(log_path, "w") as log_file:
        log_file.write("params: \n")
        log_file.write(str(params) + "\n")

    # --- Create NumPolySystem ---
    ps = NumPolySystem(n_vars=total_var_num)
    id_func = params['id']

    def v(prefix, k):
        return ps.var(id_func(prefix, k) - 1)

    eq_mask_sys = []

    # --- Initial constraints ---
    eqs, ineqs, eq_mask = get_init_constraints(
        v("x", 0) * x_max, v("y", 0) * y_max, v("rc", 0), v("rs", 0),
        v("rcld", 0), v("rsld", 0), v("rclu", 0), v("rslu", 0),
        v("rcrd", 0), v("rsrd", 0), v("rcru", 0), v("rsru", 0),
        params
    )
    for eq in eqs:
        ps.add_eq(eq)
    for ineq in ineqs:
        ps.add_ineq(ineq)
    eq_mask_sys.extend(eq_mask)
    # Bound constraints for initial x, y
    ps.add_ineq(1.0 - v("x", 0)**2)
    ps.add_ineq(1.0 - v("y", 0)**2)

    # --- Main constraint loop ---
    for k in range(1, N + 1):
        # Right finger kinematics
        eqs, ineqs, eq_mask = get_right_finger_kinematics(
            v("rcrd", k - 1), v("rcrd", k), v("rsrd", k - 1), v("rsrd", k), v("fcrd", k - 1), v("fsrd", k - 1),
            v("rcru", k - 1), v("rcru", k), v("rsru", k - 1), v("rsru", k), v("fcru", k - 1), v("fsru", k - 1),
            v("xr", k - 1) * xr_max, v("yr", k - 1) * yr_max, v("vxr", k - 1) * vxr_max, v("vyr", k - 1) * vyr_max,
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # Left finger kinematics
        eqs, ineqs, eq_mask = get_left_finger_kinematics(
            v("rcld", k - 1), v("rcld", k), v("rsld", k - 1), v("rsld", k), v("fcld", k - 1), v("fsld", k - 1),
            v("rclu", k - 1), v("rclu", k), v("rslu", k - 1), v("rslu", k), v("fclu", k - 1), v("fslu", k - 1),
            v("xl", k - 1) * xl_max, v("yl", k - 1) * yl_max, v("vxl", k - 1) * vxl_max, v("vyl", k - 1) * vyl_max,
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # Self collision info
        eqs, ineqs, eq_mask = get_self_collision_info(
            v("rcld", k), v("rsld", k), v("rclu", k), v("rslu", k),
            v("rcrd", k), v("rsrd", k), v("rcru", k), v("rsru", k),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # Right finger contact
        eqs, ineqs, eq_mask = get_right_finger_contact(
            v("x", k - 1) * x_max, v("x", k) * x_max, v("y", k - 1) * y_max, v("y", k) * y_max, v("fs", k - 1),
            v("xr", k - 1) * xr_max, v("yr", k - 1) * yr_max, v("vxr", k - 1) * vxr_max, v("vyr", k - 1) * vyr_max,
            v("dr", k - 1) * dr_max, v("vrelr", k - 1) * vrelr_max, v("lamnr", k - 1) * lamnr_max, v("lamtr", k - 1) * lamtr_max,
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # Left finger contact
        eqs, ineqs, eq_mask = get_left_finger_contact(
            v("x", k - 1) * x_max, v("x", k) * x_max, v("y", k - 1) * y_max, v("y", k) * y_max, v("fs", k - 1),
            v("xl", k - 1) * xl_max, v("yl", k - 1) * yl_max, v("vxl", k - 1) * vxl_max, v("vyl", k - 1) * vyl_max,
            v("dl", k - 1) * dl_max, v("vrell", k - 1) * vrell_max, v("lamnl", k - 1) * lamnl_max, v("lamtl", k - 1) * lamtl_max,
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # Circle dynamics
        eqs, ineqs, eq_mask = get_circle_dynamics(
            v("x", k - 1) * x_max, v("x", k) * x_max, v("y", k - 1) * y_max, v("y", k) * y_max,
            v("rc", k - 1), v("rc", k), v("rs", k - 1), v("rs", k), v("fc", k - 1), v("fs", k - 1),
            v("xl", k - 1) * xl_max, v("yl", k - 1) * yl_max, v("xr", k - 1) * xr_max, v("yr", k - 1) * yr_max,
            v("lamnl", k - 1) * lamnl_max, v("lamtl", k - 1) * lamtl_max,
            v("lamnr", k - 1) * lamnr_max, v("lamtr", k - 1) * lamtr_max,
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # Collision avoidance info
        eqs, ineqs, eq_mask = get_collision_avoidance_info(
            v("x", k) * x_max, v("y", k) * y_max,
            v("rcld", k), v("rsld", k), v("rclu", k), v("rslu", k),
            v("rcrd", k), v("rsrd", k), v("rcru", k), v("rsru", k),
            params
        )
        for eq in eqs:
            ps.add_eq(eq)
        for ineq in ineqs:
            ps.add_ineq(ineq)
        eq_mask_sys.extend(eq_mask)

        # Bound constraints
        ps.add_ineq(1.0 - v("x", k)**2)
        ps.add_ineq(1.0 - v("y", k)**2)
        ps.add_ineq(1.0 - v("xl", k - 1)**2)
        ps.add_ineq(1.0 - v("yl", k - 1)**2)
        ps.add_ineq(1.0 - v("vxl", k - 1)**2)
        ps.add_ineq(1.0 - v("vyl", k - 1)**2)
        ps.add_ineq(1.0 - v("xr", k - 1)**2)
        ps.add_ineq(1.0 - v("yr", k - 1)**2)
        ps.add_ineq(1.0 - v("vxr", k - 1)**2)
        ps.add_ineq(1.0 - v("vyr", k - 1)**2)
        ps.add_ineq(1.0 - v("dl", k - 1)**2)
        ps.add_ineq(1.0 - v("vrell", k - 1)**2)
        ps.add_ineq(1.0 - v("lamnl", k - 1)**2)
        ps.add_ineq(1.0 - v("lamtl", k - 1)**2)
        ps.add_ineq(1.0 - v("dr", k - 1)**2)
        ps.add_ineq(1.0 - v("vrelr", k - 1)**2)
        ps.add_ineq(1.0 - v("lamnr", k - 1)**2)
        ps.add_ineq(1.0 - v("lamtr", k - 1)**2)

    # --- Objective ---
    x_init_scaled = x_init / x_max
    y_init_scaled = y_init / y_max

    obj_expr = 0

    # Tracking loss
    for k in range(1, N + 1):
        th_track = loss_track_th_list[k - 1]
        loss_x = loss_track_coeff * (v("x", k) - x_init_scaled)**2
        loss_y = loss_track_coeff * (v("y", k) - y_init_scaled)**2
        loss_th = (loss_track_coeff * (v("rc", k) - np.cos(th_track))**2 +
                   loss_track_coeff * (v("rs", k) - np.sin(th_track))**2)
        obj_expr = obj_expr + loss_x + loss_y + loss_th

    # Position loss: finger position
    for k in range(1, N + 1):
        obj_expr = obj_expr + loss_pos_finger_coeff * (v("rcld", k) - np.cos(th_ld_init))**2
        obj_expr = obj_expr + loss_pos_finger_coeff * (v("rsld", k) - np.sin(th_ld_init))**2
        obj_expr = obj_expr + loss_pos_finger_coeff * (v("rclu", k) - np.cos(th_lu_init))**2
        obj_expr = obj_expr + loss_pos_finger_coeff * (v("rslu", k) - np.sin(th_lu_init))**2
        obj_expr = obj_expr + loss_pos_finger_coeff * (v("rcrd", k) - np.cos(th_rd_init))**2
        obj_expr = obj_expr + loss_pos_finger_coeff * (v("rsrd", k) - np.sin(th_rd_init))**2
        obj_expr = obj_expr + loss_pos_finger_coeff * (v("rcru", k) - np.cos(th_ru_init))**2
        obj_expr = obj_expr + loss_pos_finger_coeff * (v("rsru", k) - np.sin(th_ru_init))**2

    # Velocity loss: finger velocity
    for k in range(1, N + 1):
        obj_expr = obj_expr + loss_vel_finger_coeff * v("fsld", k - 1)**2
        obj_expr = obj_expr + loss_vel_finger_coeff * v("fslu", k - 1)**2
        obj_expr = obj_expr + loss_vel_finger_coeff * v("fsrd", k - 1)**2
        obj_expr = obj_expr + loss_vel_finger_coeff * v("fsru", k - 1)**2
        obj_expr = obj_expr + loss_vel_finger_coeff * (v("vxl", k - 1) * vxl_max)**2
        obj_expr = obj_expr + loss_vel_finger_coeff * (v("vyl", k - 1) * vyl_max)**2
        obj_expr = obj_expr + loss_vel_finger_coeff * (v("vxr", k - 1) * vxr_max)**2
        obj_expr = obj_expr + loss_vel_finger_coeff * (v("vyr", k - 1) * vyr_max)**2

    # Velocity loss: circle velocity
    for k in range(1, N + 1):
        obj_expr = obj_expr + loss_vel_circle_coeff * v("fs", k - 1)**2

    # Force loss
    for k in range(1, N + 1):
        obj_expr = obj_expr + loss_force_coeff * (v("lamnl", k - 1) * lamnl_max)**2
        obj_expr = obj_expr + loss_force_coeff * (v("lamtl", k - 1) * lamtl_max)**2
        obj_expr = obj_expr + loss_force_coeff * (v("lamnr", k - 1) * lamnr_max)**2
        obj_expr = obj_expr + loss_force_coeff * (v("lamtr", k - 1) * lamtr_max)**2

    ps.set_obj(obj_expr)

    # --- Clean polynomials ---
    ps.clean_all(tol=1e-14, if_scale=True, scale_obj=False)

    # --- Get supp_rpt data ---
    poly_data = ps.get_supp_rpt_data(kappa)

    print("Construction Finish!")

    # --- Heuristic CS clique decomposition ---
    cliques = []

    # Initial clique
    cliques.append([
        id_func("x", 0), id_func("y", 0), id_func("rc", 0), id_func("rs", 0),
        id_func("rcld", 0), id_func("rsld", 0), id_func("rclu", 0), id_func("rslu", 0),
        id_func("rcrd", 0), id_func("rsrd", 0), id_func("rcru", 0), id_func("rsru", 0),
    ])

    for k in range(1, N + 1):
        # Collision avoidance clique
        cliques.append([
            id_func("x", k), id_func("y", k),
            id_func("rcld", k), id_func("rsld", k), id_func("rclu", k), id_func("rslu", k),
            id_func("rcrd", k), id_func("rsrd", k), id_func("rcru", k), id_func("rsru", k),
        ])

        # Circle dynamics cliques
        cliques.append([
            id_func("x", k - 1), id_func("x", k), id_func("y", k - 1), id_func("y", k),
            id_func("fc", k - 1), id_func("fs", k - 1),
            id_func("xl", k - 1), id_func("yl", k - 1), id_func("xr", k - 1), id_func("yr", k - 1),
            id_func("lamnl", k - 1), id_func("lamtl", k - 1),
            id_func("lamnr", k - 1), id_func("lamtr", k - 1),
        ])
        cliques.append([
            id_func("rc", k - 1), id_func("rc", k), id_func("rs", k - 1), id_func("rs", k),
            id_func("fc", k - 1), id_func("fs", k - 1),
        ])

        # Right finger contact cliques
        cliques.append([
            id_func("x", k - 1), id_func("x", k), id_func("y", k - 1), id_func("y", k),
            id_func("fc", k - 1), id_func("fs", k - 1),
            id_func("xr", k - 1), id_func("yr", k - 1), id_func("vxr", k - 1), id_func("vyr", k - 1),
            id_func("dr", k - 1), id_func("vrelr", k - 1), id_func("lamnr", k - 1), id_func("lamtr", k - 1),
        ])
        cliques.append([
            id_func("x", k - 1), id_func("y", k - 1), id_func("xr", k - 1), id_func("yr", k - 1),
            id_func("dr", k - 1),
            id_func("rcrd", k - 1), id_func("rsrd", k - 1), id_func("rcru", k - 1), id_func("rsru", k - 1),
        ])

        # Left finger contact cliques
        cliques.append([
            id_func("x", k - 1), id_func("x", k), id_func("y", k - 1), id_func("y", k),
            id_func("fc", k - 1), id_func("fs", k - 1),
            id_func("xl", k - 1), id_func("yl", k - 1), id_func("vxl", k - 1), id_func("vyl", k - 1),
            id_func("dl", k - 1), id_func("vrell", k - 1), id_func("lamnl", k - 1), id_func("lamtl", k - 1),
        ])
        cliques.append([
            id_func("x", k - 1), id_func("y", k - 1), id_func("xl", k - 1), id_func("yl", k - 1),
            id_func("dl", k - 1),
            id_func("rcld", k - 1), id_func("rsld", k - 1), id_func("rclu", k - 1), id_func("rslu", k - 1),
        ])

        # Right finger kinematics cliques
        cliques.append([
            id_func("rcrd", k - 1), id_func("rsrd", k - 1),
            id_func("rcru", k - 1), id_func("rsru", k - 1),
            id_func("xr", k - 1), id_func("yr", k - 1),
        ])
        cliques.append([
            id_func("rcrd", k - 1), id_func("rsrd", k - 1), id_func("fcrd", k - 1), id_func("fsrd", k - 1),
            id_func("rcru", k - 1), id_func("rsru", k - 1), id_func("fcru", k - 1), id_func("fsru", k - 1),
            id_func("vxr", k - 1), id_func("vyr", k - 1),
        ])
        cliques.append([
            id_func("rcrd", k - 1), id_func("rcrd", k), id_func("rsrd", k - 1), id_func("rsrd", k),
            id_func("fcrd", k - 1), id_func("fsrd", k - 1),
            id_func("rcru", k - 1), id_func("rcru", k), id_func("rsru", k - 1), id_func("rsru", k),
            id_func("fcru", k - 1), id_func("fsru", k - 1),
        ])

        # Left finger kinematics cliques
        cliques.append([
            id_func("rcld", k - 1), id_func("rsld", k - 1),
            id_func("rclu", k - 1), id_func("rslu", k - 1),
            id_func("xl", k - 1), id_func("yl", k - 1),
        ])
        cliques.append([
            id_func("rcld", k - 1), id_func("rsld", k - 1), id_func("fcld", k - 1), id_func("fsld", k - 1),
            id_func("rclu", k - 1), id_func("rslu", k - 1), id_func("fclu", k - 1), id_func("fslu", k - 1),
            id_func("vxl", k - 1), id_func("vyl", k - 1),
        ])
        cliques.append([
            id_func("rcld", k - 1), id_func("rcld", k), id_func("rsld", k - 1), id_func("rsld", k),
            id_func("fcld", k - 1), id_func("fsld", k - 1),
            id_func("rclu", k - 1), id_func("rclu", k), id_func("rslu", k - 1), id_func("rslu", k),
            id_func("fclu", k - 1), id_func("fslu", k - 1),
        ])

    params['cliques'] = cliques

    # --- Run CSTSS ---
    start_time = time.time()
    result, res, coeff_info, aux_info = CSTSS_pybind(
        poly_data, kappa, total_var_num, params
    )
    elapsed_time = time.time() - start_time
    aux_info["result"] = result
    params["aux_info"] = aux_info

    with open(log_path, "a") as log_file:
        log_file.write(f"PlanarHand  N={N}, Relax={relax_mode}, TS={ts_mode}, CS={cs_mode}, "
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
    clique_supp_list = []
    clique_coeff_list = []
    kappa_width = 2 * kappa
    if "cliques" in aux_info and aux_info["cliques"]:
        cliques_rank = params["cliques_rank"]
        for i in range(len(cliques_rank)):
            ii = cliques_rank[i]
            sorted_vars = sorted(aux_info["cliques"][ii])
            supp = np.zeros((len(sorted_vars), kappa_width), dtype=np.float64)
            for idx_v, j in enumerate(sorted_vars):
                supp[idx_v, -1] = j
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

    params['self_cliques'] = cliques

    if not params['if_solve']:
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
        params_to_save = {k: vv for k, vv in params.items() if not callable(vv)}
        with open("./data/" + prefix_str + "params.pkl", "wb") as f:
            pickle.dump(params_to_save, f)
        return

    # --- Extract solution ---
    Xopt = res['Xopt']
    yopt = res['yopt']
    Sopt = res['Sopt']

    if relax_mode == 'MOMENT':
        Xs = Xopt
    elif relax_mode == 'SOS':
        Xs = [-S for S in Sopt]

    ts_info = aux_info["ts_info"]
    cliques_aux = aux_info["cliques"]
    mon_rpt = aux_info["mon_rpt"]

    mom_mat_num = sum(len(ts_info[i]) for i in range(len(cliques_aux)))
    mom_mat_rpt = [None] * mom_mat_num

    idx = 0
    for i in range(len(cliques_aux)):
        for j in range(len(ts_info[i])):
            rpt = mon_rpt[i][ts_info[i][j], :]
            rpt = np.hstack([np.zeros_like(rpt), rpt])
            mom_mat_rpt[idx] = rpt
            idx += 1

    if ts_mode == "NON":
        v_opt_robust, output_info_robust = robust_extract_CS(Xs, mom_mat_rpt, total_var_num, 1e-2)
        with open("./data/" + prefix_str + "v_opt_robust.pkl", "wb") as f:
            pickle.dump(v_opt_robust, f)
        v_opt_ordered, output_info_ordered = ordered_extract_CS(
            Xs, mom_mat_rpt, total_var_num, 1e-2, params.get("cliques_rank", []))
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
    params_to_save = {k: vv for k, vv in params.items() if not callable(vv)}
    with open("./data/" + prefix_str + "params.pkl", "wb") as f:
        pickle.dump(params_to_save, f)

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.5f} s")
    with open(log_path, "a") as log_file:
        log_file.write(f"total time={total_time:.5f}\n")


if __name__ == "__main__":
    main()
