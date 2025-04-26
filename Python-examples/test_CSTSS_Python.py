import time
from sympy import symbols, total_degree, Poly
import numpy as np

import sys 
import os
# Add the parent directory to sys.path so Python can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
from SPOT.PYTHON.naive_extract import naive_extract

if __name__ == "__main__":
    x1, x2, x3 = symbols('x1 x2 x3')
    x = [x1, x2, x3]

    # all monomials degree <= 3
    f = x1 + x2 + x3
    g = [2 - x1, 2 - x2, 2 - x3]
    h = [x1**2 + x2**2 - 1, x2**2 + x3**2 - 1, x2 - 0.5]
    d = 2

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

    result, res, coeff_info, aux_info = CSTSS_pybind(f, g, h, d, x, input_info)

    # print(aux_info['cliques'])

    if relax_mode == 'MOMENT':
        v_opt, output_info = naive_extract(res['Xopt'], aux_info['mon_rpt'], aux_info['ts_info'], len(x))

    print(v_opt)






    