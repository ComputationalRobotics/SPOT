import numpy as np

import sys
import os
# Add the parent directory to sys.path so Python can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
from SPOT.PYTHON.numpoly import NumPolySystem, NumPolyExpr
from SPOT.PYTHON.naive_extract import naive_extract

if __name__ == "__main__":
    ps = NumPolySystem(n_vars=3)
    x1 = ps.var(0)
    x2 = ps.var(1)
    x3 = ps.var(2)

    # all monomials degree <= 3
    f = x1 + x2 + x3
    g = [NumPolyExpr.from_const(2) - x1, NumPolyExpr.from_const(2) - x2, NumPolyExpr.from_const(2) - x3]
    h = [x1**2 + x2**2 - NumPolyExpr.from_const(1), x2**2 + x3**2 - NumPolyExpr.from_const(1), x2 - NumPolyExpr.from_const(0.5)]
    d = 2

    ps.set_obj(f)
    for gi in g:
        ps.add_ineq(gi)
    for hi in h:
        ps.add_eq(hi)

    ps.clean_all(tol=1e-14, if_scale=True, scale_obj=False)
    poly_data = ps.get_supp_rpt_data(d)

    relax_mode = 'MOMENT'
    cs_mode = 'NON'
    ts_mode = 'MD'
    ts_mom_mode = 'NON'
    ts_eq_mode = 'NON'
    # cs_cliques = []
    cs_cliques = [np.array([1, 2, 3])]
    if_solve = True

    input_info = dict()
    input_info['relax_mode'] = relax_mode
    input_info['cs_mode'] = cs_mode
    input_info['ts_mode'] = ts_mode
    input_info['ts_mom_mode'] = ts_mom_mode
    input_info['ts_eq_mode'] = ts_eq_mode
    input_info['if_solve'] = if_solve
    input_info['cliques'] = cs_cliques

    result, res, coeff_info, aux_info = CSTSS_pybind(poly_data, d, 3, input_info)

    # print(aux_info['cliques'])

    if relax_mode == 'MOMENT':
        v_opt, output_info = naive_extract(res['Xopt'], aux_info['mon_rpt'], aux_info['ts_info'], 3)

    print(v_opt)
