
import time
from sympy import symbols, total_degree, Poly
import numpy as np

from SPOT.PYTHON.mosek_standard_sdp_test_1 import mosek_standard_sdp_test_1
from SPOT.PYTHON.mosek_standard_sdp_test_2 import mosek_standard_sdp_test_2

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'build/'))
import CSTSS_Python_pybind


def CSTSS_pybind(poly_data, d, n, input_info):
    """
    Optimized CSTSS solver using low-level MOSEK API.

    This version bypasses MOSEK Fusion API's expression building overhead
    by using the low-level API directly, similar to MATLAB's approach.

    Args:
        poly_data: Dictionary with polynomial data
        d: Relaxation order (kappa)
        n: Number of variables
        input_info: Solver parameters
        use_v2: If True, use the even more optimized V2 version

    Returns:
        Same as CSTSS_pybind_numpoly
    """
    relax_mode = input_info['relax_mode']
    cs_mode = input_info['cs_mode']
    cs_cliques = input_info['cliques']
    ts_mode = input_info['ts_mode']
    ts_mom_mode = input_info['ts_mom_mode']
    ts_eq_mode = input_info['ts_eq_mode']
    if_solve = input_info['if_solve']
    kappa = d

    ts_cliques = []

    # Use pre-computed polynomial data directly
    aux_info = {
        'supp_rpt_f': poly_data['supp_rpt_f'],
        'supp_rpt_g': poly_data['supp_rpt_g'],
        'supp_rpt_h': poly_data['supp_rpt_h'],
        'coeff_f': poly_data['coeff_f'],
        'coeff_g': poly_data['coeff_g'],
        'coeff_h': poly_data['coeff_h'],
        'dj_g': poly_data['dj_g'],
        'dj_h': poly_data['dj_h'],
    }

    start_time = time.time()
    cs_info, ts_info, moment_info, sos_info = CSTSS_Python_pybind.my_function(
        kappa, n,
        aux_info['coeff_f'], aux_info['supp_rpt_f'],
        aux_info['coeff_g'], aux_info['supp_rpt_g'],
        aux_info['coeff_h'], aux_info['supp_rpt_h'],
        aux_info['dj_g'], aux_info['dj_h'],
        relax_mode,
        cs_mode,
        ts_mode,
        ts_mom_mode,
        ts_eq_mode,
        cs_cliques,
        ts_cliques
    )
    CSTSS_time = time.time() - start_time

    coeff_info = dict()
    if if_solve:
        if relax_mode == 'MOMENT':
            result, res, solve_time = mosek_standard_sdp_test_1(
                moment_info.A_moment,
                moment_info.C_moment,
                moment_info.b_moment,
                ts_info.tI_size
            )
            coeff_info['A'] = moment_info.A_moment
            coeff_info['C'] = moment_info.C_moment
            coeff_info['b'] = moment_info.b_moment

        elif relax_mode == 'SOS':
            result, res, solve_time = mosek_standard_sdp_test_2(
                sos_info.A_sos,
                sos_info.a_sos,
                sos_info.b_sos,
                sos_info.c_sos,
                ts_info.tI_size
            )

            coeff_info['A'] = sos_info.A_sos
            coeff_info['prob_a'] = sos_info.a_sos
            coeff_info['b'] = sos_info.b_sos
            coeff_info['prob_c'] = sos_info.c_sos
    else:
        result = []
        res = []
        coeff_info = []
        solve_time = 0

    aux_info['mosek_time'] = solve_time
    aux_info['time'] = CSTSS_time
    aux_info['cliques'] = cs_info.cI
    aux_info['clique_size'] = ts_info.tI_size
    aux_info['ts_info'] = ts_info.tI
    aux_info['mon_rpt'] = cs_info.mon
    aux_info['mon_rpt_g'] = cs_info.mon_g
    aux_info['mon_rpt_h'] = cs_info.mon_h

    return result, res, coeff_info, aux_info
