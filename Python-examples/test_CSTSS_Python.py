import numpy as np

import sys
import os
# Add the parent directory to sys.path so Python can find the package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from SPOT.PYTHON.CSTSS_pybind import CSTSS_pybind
from SPOT.PYTHON.numpoly import NumPolySystem, NumPolyExpr
from SPOT.PYTHON.naive_extract import naive_extract

if __name__ == "__main__":
    # --- Step 1: Create a polynomial system ---
    # NumPolySystem manages variables and collects objective/constraints.
    # n_vars is the total number of decision variables (here x1, x2, x3).
    ps = NumPolySystem(n_vars=3)

    # --- Step 2: Declare variables ---
    # ps.var(i) returns a NumPolyExpr representing the (i+1)-th variable.
    # Indexing is 0-based: ps.var(0) -> x1, ps.var(1) -> x2, ps.var(2) -> x3.
    x1 = ps.var(0)
    x2 = ps.var(1)
    x3 = ps.var(2)

    # --- Step 3: Build polynomial expressions ---
    # Supports standard Python arithmetic: +, -, *, **.
    # Objective: minimize f = x1 + x2 + x3
    f = x1 + x2 + x3

    # Inequality constraints (g >= 0): 2 - xi >= 0  =>  xi <= 2
    g = [2 - x1, 2 - x2, 2 - x3]

    # Equality constraints (h = 0):
    #   x1^2 + x2^2 = 1,  x2^2 + x3^2 = 1,  x2 = 0.5
    h = [x1**2 + x2**2 - 1, x2**2 + x3**2 - 1, x2 - 0.5]

    # d = relaxation order (kappa); determines the SDP size (moment matrix order)
    d = 2

    # --- Step 4: Register objective and constraints ---
    # ps.set_obj(f)       -- set the polynomial objective to minimize
    # ps.add_ineq(g)      -- add g >= 0 inequality constraint
    # ps.add_eq(h)        -- add h == 0 equality constraint
    ps.set_obj(f)
    for gi in g:
        ps.add_ineq(gi)
    for hi in h:
        ps.add_eq(hi)

    # --- Step 5: Clean all polynomials ---
    # Removes near-zero coefficients (|c| < tol) and optionally scales each
    # constraint so its largest coefficient is 1 (if_scale=True).
    # scale_obj=False keeps the objective unscaled so its optimal value is preserved.
    ps.clean_all(tol=1e-14, if_scale=True, scale_obj=False)

    # --- Step 6: Convert to supp_rpt format for CSTSS ---
    # get_supp_rpt_data(d) packs all polynomials into the sparse support/coefficient
    # representation expected by the CSTSS solver.
    # Returns a dict with keys: supp_rpt_f, coeff_f, supp_rpt_g, coeff_g,
    #                           supp_rpt_h, coeff_h, dj_g, dj_h
    poly_data = ps.get_supp_rpt_data(d)

    # --- Step 7: Configure the CSTSS solver ---
    # relax_mode : 'MOMENT' uses the moment-SOS hierarchy (recommended);
    #              alternative is 'SOS' for a pure SOS relaxation.
    relax_mode = 'MOMENT'

    # cs_mode    : correlative sparsity mode.
    #              'NON' = no correlative sparsity (dense);
    #              'CS'  = exploit correlative sparsity with auto-detected cliques.
    cs_mode = 'NON'

    # ts_mode    : term sparsity mode applied to the moment matrix.
    #              'MD'  = monomial-diagonal term sparsity;
    #              'NON' = no term sparsity.
    ts_mode = 'MD'

    # ts_mom_mode / ts_eq_mode : term sparsity for moment / equality blocks.
    #              'NON' = disabled.
    ts_mom_mode = 'NON'
    ts_eq_mode = 'NON'

    # cs_cliques : list of variable-index arrays defining correlative-sparsity cliques.
    #              Each array contains 1-indexed variable IDs belonging to one clique.
    #              Use [] (empty list) to let the solver auto-detect cliques when cs_mode='CS'.
    # cs_cliques = []
    cs_cliques = [np.array([1, 2, 3])]

    # if_solve   : True  = assemble and solve the SDP;
    #              False = only build the data structures without calling the solver.
    if_solve = True

    input_info = dict()
    input_info['relax_mode'] = relax_mode
    input_info['cs_mode'] = cs_mode
    input_info['ts_mode'] = ts_mode
    input_info['ts_mom_mode'] = ts_mom_mode
    input_info['ts_eq_mode'] = ts_eq_mode
    input_info['if_solve'] = if_solve
    input_info['cliques'] = cs_cliques

    # --- Step 8: Run CSTSS ---
    # Arguments: poly_data (supp_rpt dict), d (relaxation order),
    #            n_vars (number of variables), input_info (solver config).
    # Returns:
    #   result     -- solver status / summary
    #   res        -- raw SDP result (Xopt, etc.)
    #   coeff_info -- coefficient data for the relaxation
    #   aux_info   -- auxiliary info: cliques, monomial support (mon_rpt), ts_info, etc.
    result, res, coeff_info, aux_info = CSTSS_pybind(poly_data, d, 3, input_info)

    # print(aux_info['cliques'])

    # --- Step 9: Extract optimal solution from the moment matrix ---
    # naive_extract recovers candidate solutions from the SDP optimal moment matrix Xopt.
    # Arguments: Xopt         -- optimal moment matrix from the SDP
    #            mon_rpt      -- monomial support in supp_rpt format
    #            ts_info      -- term sparsity block structure
    #            n_vars       -- number of variables
    # Returns:  v_opt        -- array of recovered optimal variable values
    #           output_info  -- additional extraction details (residuals, rank, etc.)
    if relax_mode == 'MOMENT':
        v_opt, output_info = naive_extract(res['Xopt'], aux_info['mon_rpt'], aux_info['ts_info'], 3)

    print(v_opt)
