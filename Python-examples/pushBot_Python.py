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
    # a: 0 : N+1 --> id: 1 : N+2
    # rc: 0 : N --> id: N+2 + (1 : N+1)
    # rs: 0 : N+1 --> id: 2*N+3 + (1 : N+2)
    # fc: 0 : N --> id: 3*N+5 + (1 : N+1)
    # fs: 0 : N --> id: 4*N+6 + (1 : N+1)
    # u: 1 : N --> id: 5*N+7 + (1 : N)
    # lam1: 1 : N --> id: 6*N+7 + (1 : N)
    # lam2: 1 : N --> id: 7*N+7 + (1 : N)
    if prefix == "a":
        return k + 1
    elif prefix == "rc":
        return N + 2 + k + 1
    elif prefix == "rs":
        return 2 * N + 3 + k + 1
    elif prefix == "fc":
        return 3 * N + 5 + k + 1
    elif prefix == "fs":
        return 4 * N + 6 + k + 1
    elif prefix == "u":
        return 5 * N + 7 + k
    elif prefix == "lam1":
        return 6 * N + 7 + k
    elif prefix == "lam2":
        return 7 * N + 7 + k
    else:
        raise ValueError("Unknown prefix")


def get_var_mapping(params):
    id_func = params['id']
    N = params['N']
    var_mapping = {}
    for k in range(N + 2):
        var_mapping[id_func("a", k)] = f"a_{{{k}}}"
        var_mapping[id_func("rs", k)] = f"r_{{s, {k}}}"
    for k in range(N + 1):
        var_mapping[id_func("rc", k)] = f"r_{{c, {k}}}"
        var_mapping[id_func("fc", k)] = f"f_{{c, {k}}}"
        var_mapping[id_func("fs", k)] = f"f_{{s, {k}}}"
    for k in range(1, N + 1):
        var_mapping[id_func("u", k)] = f"u_{{{k}}}"
        var_mapping[id_func("lam1", k)] = f"\\lambda_{{1, {k}}}"
        var_mapping[id_func("lam2", k)] = f"\\lambda_{{2, {k}}}"
    return var_mapping


def get_remapped_ids(params):
    N = params['N']
    total_var_num = params['total_var_num']
    ids_remap = np.zeros(total_var_num, dtype=int)
    idx = 1
    for k in range(N + 2):
        ids_remap[get_id("a", k, N) - 1] = idx; idx += 1
        ids_remap[get_id("rs", k, N) - 1] = idx; idx += 1
        if k < N + 1:
            ids_remap[get_id("rc", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("fc", k, N) - 1] = idx; idx += 1
            ids_remap[get_id("fs", k, N) - 1] = idx; idx += 1
            if k > 0:
                ids_remap[get_id("u", k, N) - 1] = idx; idx += 1
                ids_remap[get_id("lam1", k, N) - 1] = idx; idx += 1
                ids_remap[get_id("lam2", k, N) - 1] = idx; idx += 1
    return ids_remap


def get_eq_init(a_0, a_1, rc_0, rs_0, rs_1, fc_0, fs_0,
                a_init, v_init, th_init, dth_init,
                dt, mc, mp, l, g, d1, d2, k1, k2, fc_min):
    eq1 = a_0 - a_init
    eq2 = a_1 - (a_init + dt * v_init)
    eq3 = rc_0 - np.cos(th_init)
    eq4 = rs_0 - np.sin(th_init)
    eq5 = fc_0 - np.cos(dt * dth_init)
    eq6 = fs_0 - np.sin(dt * dth_init)
    eq7 = rs_1 - (np.sin(th_init) * np.cos(dt * dth_init) + np.cos(th_init) * np.sin(dt * dth_init))
    eq8 = rc_0**2 + rs_0**2 - 1
    eq9 = fc_0**2 + fs_0**2 - 1
    eqs = [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8, eq9]
    eq_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0]
    return eqs, eq_mask


def get_eq_dyn(a_km1, a_k, a_kp1,
               rc_km1, rc_k, rs_km1, rs_k, rs_kp1, fc_km1, fc_k, fs_km1, fs_k,
               u_k, lam1_k, lam2_k,
               dt, mc, mp, l, g, d1, d2, k1, k2, fc_min):
    eq1 = (mc + mp) * (a_km1 - 2 * a_k + a_kp1) / (dt**2) + mp * l * (rs_km1 - 2 * rs_k + rs_kp1) / (dt**2) - (u_k + lam1_k - lam2_k)
    eq2 = l * (fs_k - fs_km1) / (dt**2) + ((a_km1 - 2 * a_k + a_kp1) / (dt**2) + lam2_k - lam1_k) * rc_k + g * rs_k
    eq3 = lam1_k * (lam1_k / k1 + d1 + a_k + l * rs_k)
    eq4 = lam2_k * (lam2_k / k2 + d2 - a_k - l * rs_k)
    eq5 = rc_k - (rc_km1 * fc_km1 - rs_km1 * fs_km1)
    eq6 = rs_kp1 - (rs_k * fc_k + rc_k * fs_k)
    eq7 = rc_k**2 + rs_k**2 - 1
    eq8 = fc_k**2 + fs_k**2 - 1
    eqs = [eq1, eq2, eq3, eq4, eq5, eq6, eq7, eq8]
    eq_mask = [1, 1, 1, 1, 1, 1, 0, 1]
    return eqs, eq_mask


def get_ineq_dyn(a_km1, a_k, a_kp1,
                 rc_km1, rc_k, rs_km1, rs_k, rs_kp1, fc_km1, fc_k, fs_km1, fs_k,
                 u_k, lam1_k, lam2_k,
                 dt, mc, mp, l, g, d1, d2, k1, k2, fc_min):
    ineq1 = lam1_k
    ineq2 = lam1_k / k1 + d1 + a_k + l * rs_k
    ineq3 = lam2_k
    ineq4 = lam2_k / k2 + d2 - a_k - l * rs_k
    ineq5 = fc_k - fc_min
    return [ineq1, ineq2, ineq3, ineq4, ineq5]


####################
# Main Application #
####################

def main():
    total_start = time.time()

    # --- CSTSS parameters ---
    params = {}
    kappa = 2; params['kappa'] = kappa
    relax_mode = "SOS"; params['relax_mode'] = relax_mode
    cs_mode = "MD"; params['cs_mode'] = cs_mode
    ts_mode = "NON"; params['ts_mode'] = ts_mode
    ts_mom_mode = "NON"; params['ts_mom_mode'] = ts_mom_mode
    ts_eq_mode = "NON"; params['ts_eq_mode'] = ts_eq_mode
    if_solve = True; params['if_solve'] = if_solve
    if_mex = True; params['if_mex'] = if_mex

    # --- Hyper-parameters ---
    N = 30; params['N'] = N
    dt = 0.05; params['dt'] = dt
    mc = 1.0; params['mc'] = mc
    mp = 0.1; params['mp'] = mp
    l = 0.8; params['l'] = l
    g = 9.8; params['g'] = g
    d1 = 1.0; params['d1'] = d1
    d2 = 1.0; params['d2'] = d2
    k1 = 200.0; params['k1'] = k1
    k2 = 200.0; params['k2'] = k2

    # --- Variable bounds ---
    a_max = 1.0; params['a_max'] = a_max
    u_max = 20.0; params['u_max'] = u_max
    lam1_max = 100.0; params['lam1_max'] = lam1_max
    lam2_max = 100.0; params['lam2_max'] = lam2_max
    fc_min = 0.7; params['fc_min'] = fc_min

    # --- Initial states ---
    a_init = 0.9; params['a_init'] = a_init
    v_init = 0.0; params['v_init'] = v_init
    th_init = 0.0; params['th_init'] = th_init
    dth_init = 0.0; params['dth_init'] = dth_init

    # --- Final states ---
    a_final = 0.0; params['a_final'] = a_final
    v_final = 0.0; params['v_final'] = v_final
    th_final = np.pi; params['th_final'] = th_final
    dth_final = 0.0; params['dth_final'] = dth_final

    # --- Objective coefficients ---
    a_coeff = 1.0; params['a_coeff'] = a_coeff
    th_coeff = 1.0; params['th_coeff'] = th_coeff
    dth_coeff = 1.0; params['dth_coeff'] = dth_coeff
    u_coeff = 1.0; params['u_coeff'] = u_coeff
    final_penalty = 10.0; params['final_penalty'] = final_penalty

    # --- Total variable number ---
    total_var_num = 8 * N + 7; params['total_var_num'] = total_var_num
    params['id'] = lambda prefix, k: get_id(prefix, k, N)
    var_mapping = get_var_mapping(params); params['var_mapping'] = var_mapping

    # --- Get remapping information ---
    ids_remap = get_remapped_ids(params); params['ids_remap'] = ids_remap

    # --- File management ---
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    prefix_str = "PushBot_Python_Optimized/" + current_time + "/"
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
        v("a", 0) * a_max, v("a", 1) * a_max,
        v("rc", 0), v("rs", 0), v("rs", 1), v("fc", 0), v("fs", 0),
        a_init, v_init, th_init, dth_init,
        dt, mc, mp, l, g, d1, d2, k1, k2, fc_min
    )
    for eq in eqs:
        ps.add_eq(eq)
    eq_mask_sys.extend(eq_mask)

    # Dynamics equality constraints
    for k in range(1, N + 1):
        eqs, eq_mask = get_eq_dyn(
            v("a", k - 1) * a_max, v("a", k) * a_max, v("a", k + 1) * a_max,
            v("rc", k - 1), v("rc", k),
            v("rs", k - 1), v("rs", k), v("rs", k + 1),
            v("fc", k - 1), v("fc", k),
            v("fs", k - 1), v("fs", k),
            v("u", k) * u_max, v("lam1", k) * lam1_max, v("lam2", k) * lam2_max,
            dt, mc, mp, l, g, d1, d2, k1, k2, fc_min
        )
        for eq in eqs:
            ps.add_eq(eq)
        eq_mask_sys.extend(eq_mask)

    # Dynamics inequality constraints
    for k in range(1, N + 1):
        ineqs = get_ineq_dyn(
            v("a", k - 1) * a_max, v("a", k) * a_max, v("a", k + 1) * a_max,
            v("rc", k - 1), v("rc", k),
            v("rs", k - 1), v("rs", k), v("rs", k + 1),
            v("fc", k - 1), v("fc", k),
            v("fs", k - 1), v("fs", k),
            v("u", k) * u_max, v("lam1", k) * lam1_max, v("lam2", k) * lam2_max,
            dt, mc, mp, l, g, d1, d2, k1, k2, fc_min
        )
        for ineq in ineqs:
            ps.add_ineq(ineq)

        # Bounds on variables
        if k == 1:
            ps.add_ineq(1 - v("a", k - 1)**2)
            ps.add_ineq(1 - v("a", k)**2)
            ps.add_ineq(1 - v("rc", k - 1)**2 - v("rs", k - 1)**2)
            ps.add_ineq(1 - v("fc", k - 1)**2 - v("fs", k - 1)**2)
        ps.add_ineq(1 - v("a", k + 1)**2)
        ps.add_ineq(1 - v("rs", k + 1)**2)
        ps.add_ineq(1 - v("u", k)**2)
        ps.add_ineq(1 - v("lam1", k)**2)
        ps.add_ineq(1 - v("lam2", k)**2)

    # --- Objective ---
    a_final_scaled = a_final / a_max
    rc_final = np.cos(th_final)
    rs_final = np.sin(th_final)
    fc_final = np.cos(dt * dth_final)
    fs_final = np.sin(dt * dth_final)

    obj_expr = NumPolyExpr.from_const(0)
    for k in range(1, N + 1):
        if k < N:
            expr = (a_coeff * (v("a", k + 1) - a_final_scaled)**2 +
                    th_coeff * (v("rc", k) - rc_final)**2 +
                    th_coeff * (v("rs", k + 1) - rs_final)**2 +
                    dth_coeff * (v("fc", k) - fc_final)**2 +
                    dth_coeff * (v("fs", k) - fs_final)**2 +
                    u_coeff * v("u", k)**2)
            obj_expr = obj_expr + expr
        else:
            expr = (final_penalty * a_coeff * (v("a", k + 1) - a_final_scaled)**2 +
                    final_penalty * th_coeff * (v("rc", k) - rc_final)**2 +
                    final_penalty * th_coeff * (v("rs", k + 1) - rs_final)**2 +
                    final_penalty * dth_coeff * (v("fc", k) - fc_final)**2 +
                    final_penalty * dth_coeff * (v("fs", k) - fs_final)**2 +
                    u_coeff * v("u", k)**2)
            obj_expr = obj_expr + expr

    ps.set_obj(obj_expr)

    # --- Clean polynomials ---
    ps.clean_all(tol=1e-14, if_scale=True, scale_obj=False)

    # --- Get supp_rpt data ---
    poly_data = ps.get_supp_rpt_data(kappa)

    # --- Initialize cliques ---
    cliques = []
    for k in range(1, N + 1):
        clique = [
            params['id']("a", k - 1), params['id']("a", k), params['id']("a", k + 1),
            params['id']("rc", k - 1), params['id']("rc", k),
            params['id']("rs", k - 1), params['id']("rs", k), params['id']("rs", k + 1),
            params['id']("fc", k - 1), params['id']("fc", k),
            params['id']("fs", k - 1), params['id']("fs", k),
            params['id']("u", k),
            params['id']("lam1", k), params['id']("lam2", k),
        ]
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

    with open(log_path, "a") as log_file:
        log_file.write(f"PushBot  N={N}, Relax={relax_mode}, TS={ts_mode}, CS={cs_mode}, "
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

    if not params['if_solve']:
        params['self_cliques'] = cliques
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
    params['self_cliques'] = cliques

    total_time = time.time() - total_start
    print(f"\nTotal time: {total_time:.5f} s")
    with open(log_path, "a") as log_file:
        log_file.write(f"total time={total_time:.5f}\n")


if __name__ == "__main__":
    main()
