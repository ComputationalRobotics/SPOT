import time
from sympy import symbols, total_degree, Poly
import numpy as np

from SPOT.PYTHON.mosek_standard_sdp_test_1 import mosek_standard_sdp_test_1
from SPOT.PYTHON.mosek_standard_sdp_test_2 import mosek_standard_sdp_test_2

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'build/'))
import CSTSS_Python_pybind

def supp_rpt(f, x, d):
    poly = Poly(f, x)
    # ordinary monomials presentation
    monomials = poly.monoms()
    coeff = poly.coeffs()

    # reset order based on matlab version
    def sort_key(monomial):
        degree = sum(monomial)
        return (degree, [-exp for exp in monomial])

    sorted_indices = sorted(range(len(monomials)), key=lambda i: sort_key(monomials[i]))
    sorted_monomials = [monomials[i] for i in sorted_indices]
    sorted_coeff = [coeff[i] for i in sorted_indices]

    # x1->1, x3->3
    x_vars_index = {var: i + 1 for i, var in enumerate(x)}
    seqs = np.zeros((len(sorted_monomials), d), dtype=int)

    for i, monomial in enumerate(sorted_monomials):
        current_seq = []
        for var_idx, exp in enumerate(monomial):
            current_seq.extend([x_vars_index[x[var_idx]]] * exp)
        if current_seq:
            seqs[i, -len(current_seq):] = current_seq

    seqs = seqs.astype(np.float64)
    sorted_coeff = np.asanyarray(sorted_coeff)
    sorted_coeff = sorted_coeff.astype(np.float64)

    return seqs, sorted_coeff

def process_polynomials(f, g, h, x, d):
    # record time example
    ticp = time.time()

    m_ineq = len(g)
    m_eq = len(h)

    dj_g = np.zeros(m_ineq)
    dj_h = np.zeros(m_eq)
    supp_rpt_g = [None] * m_ineq
    supp_rpt_h = [None] * m_eq
    coeff_g = [None] * m_ineq
    coeff_h = [None] * m_eq

    supp_rpt_f, coeff_f = supp_rpt(f, x, 2 * d)

    for i in range(m_ineq):
        dj_g[i] = np.ceil(total_degree(g[i]) / 2)
        supp_rpt_g[i], coeff_g[i] = supp_rpt(g[i], x, 2 * d)

    for i in range(m_eq):
        dj_h[i] = total_degree(h[i])
        supp_rpt_h[i], coeff_h[i] = supp_rpt(h[i], x, 2 * d)

    elapsed_time = time.time() - ticp
    print(f"Parameter time: {elapsed_time:.4f} seconds")

    aux_info = {
        'supp_rpt_f': supp_rpt_f,
        'supp_rpt_g': supp_rpt_g,
        'supp_rpt_h': supp_rpt_h,
        'coeff_f': coeff_f,
        'coeff_g': coeff_g,
        'coeff_h': coeff_h,
        'dj_g': dj_g,
        'dj_h': dj_h,
    }

    return aux_info

def CSTSS_pybind(f, g, h, d, x, input_info):
    relax_mode = input_info['relax_mode']
    cs_mode = input_info['cs_mode']
    cs_cliques = input_info['cliques']
    ts_mode = input_info['ts_mode'] 
    ts_mom_mode = input_info['ts_mom_mode'] 
    ts_eq_mode = input_info['ts_eq_mode']
    if_solve = input_info['if_solve']
    kappa = d 

    ts_cliques = []

    aux_info = process_polynomials(f, g, h, x, d)
    n = len(x)

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
            result, res, solve_time = mosek_standard_sdp_test_1(moment_info.A_moment, moment_info.C_moment, moment_info.b_moment, ts_info.tI_size)
            coeff_info['A'] = moment_info.A_moment
            coeff_info['C'] = moment_info.C_moment
            coeff_info['b'] = moment_info.b_moment
        elif relax_mode == 'SOS':
            result, res, solve_time = mosek_standard_sdp_test_2(sos_info.A_sos, sos_info.a_sos, sos_info.b_sos, sos_info.c_sos, ts_info.tI_size)
            coeff_info['A'] = sos_info.A_sos
            coeff_info['prob_a'] = sos_info.a_sos
            coeff_info['b'] = sos_info.b_sos
            coeff_info['prob_c'] = sos_info.c_sos
    else: 
        result = []
        res = []
        coeff_info = []
    
    aux_info['mosek_time'] = solve_time
    aux_info['time'] = CSTSS_time 
    aux_info['cliques'] = cs_info.cI
    aux_info['clique_size'] = ts_info.tI_size
    aux_info['ts_info'] = ts_info.tI
    aux_info['mon_rpt'] = cs_info.mon
    aux_info['mon_rpt_g'] = cs_info.mon_g
    aux_info['mon_rpt_h'] = cs_info.mon_h

    return result, res, coeff_info, aux_info