import time
import numpy as np
import mosek
import sys
from mosek.fusion import Model, Domain, Expr, ObjectiveSense

def mosek_standard_sdp_test_2(A, prob_a, b, prob_c, s):
    """
    A        : numpy array of shape (numBarCoeff, 5).
               Each row [i, j, l, k, val] indicates that the entry (k,l)
               of the j-th PSD block is multiplied by 'val' in the i-th constraint.
               NOTE: In MATLAB, these are typically 1-based indices. You may need
               to subtract 1 if the data is coming directly from MATLAB.

    prob_a   : numpy array of shape (nnzLin, 3).
               Each row [rowIdx, colIdx, val] indicates that x[colIdx] is
               multiplied by 'val' in the rowIdx-th linear constraint.
               Again, watch out for 1-based vs. 0-based indexing.

    b        : 1D array-like of right-hand side values, length = number of constraints.

    prob_c   : 1D array-like of length = number of linear variables, i.e. the
               objective coefficient vector c.

    s        : array-like of block dimensions for the PSD variables.
               Example: s = [3, 5] means you have two PSD blocks X0 in S^3, X1 in S^5.

    Returns
    -------
    M                : The Fusion Model object (holding the solution).
    mosek_time       : The elapsed time for M.solve().
    iter             : Number of interior-point iterations (if available).
    pfeas, dfeas     : Approximate primal/dual feasibility (if available).
    max_residual     : A placeholder for a "residual measure" (Fusion does not
                       provide exactly the same info as 'get_mosek_gap' in the
                       MATLAB low-level API).  You can retrieve other residual
                       data via M.getSolverDoubleInfo(...).
    """

    start_time = time.time()
    # Create the model
    M = Model("mosek_standard_sdp_test_2")

    # (Optional) Turn on detailed logging
    M.setSolverParam("log", 1)
    
    # Send log output to stdout
    M.setLogHandler(sys.stdout)

    #
    # 1) Create linear variables, x in R^{len(prob_c)}.
    #
    prob_c = prob_c.astype(float)
    n = len(prob_c)
    x = M.variable("x", n, Domain.unbounded())

    #
    # 2) Create PSD (bar) variables according to dimensions in s.
    #    If s = [s1, s2, ...], we create a list of PSD blocks X[0], X[1], etc.
    #
    Xblocks = []
    for idx, dim in enumerate(s):
        Xblocks.append(M.variable(f"X_{idx}", [dim, dim], Domain.inPSDCone()))

    #
    # 3) Build up the linear constraints:  A*x + sum of PSD-terms == b
    #
    #    We'll accumulate an expression for each constraint row in a list,
    #    then constrain it to equal b[row].
    #
    b = np.squeeze(b)
    m = len(b)  # Number of constraints
    # exprs = [Expr.constTerm(0.0) for _ in range(m)]

    row_terms = [[] for _ in range(m)]

    for rowIdx, colIdx, val in prob_a:
        rowIdx  = int(rowIdx)   
        colIdx  = int(colIdx)
        row_terms[rowIdx].append(Expr.mul(val, x.index(int(colIdx))))

    for subi, subj, subl, subk, val in A:
        subi = int(subi)
        subj = int(subj)
        subl = int(subl)
        subk = int(subk)
        t = Expr.mul(val, Xblocks[int(subj)].index(int(subk), int(subl)))
        row_terms[int(subi)].append(t)
        if subk != subl:                      # symmetric counterpart
            t2 = Expr.mul(val, Xblocks[int(subj)].index(int(subl), int(subk)))
            row_terms[int(subi)].append(t2)

    exprs = [Expr.add(terms) if terms else Expr.constTerm(0.0)
            for terms in row_terms]

    # # 3a) Add the linear part: prob_a has [rowIdx, colIdx, val].
    # #     For each row, the contribution is val * x[colIdx].
    # for (rowIdx, colIdx, val) in prob_a:
    #     rowIdx  = int(rowIdx)   # ensure Python integer
    #     colIdx  = int(colIdx)
    #     exprs[rowIdx] = Expr.add(exprs[rowIdx],
    #                              Expr.mul(val, x.index(colIdx)))

    # # 3b) Add the SDP (bar) part: A has [subi, subj, subl, subk, val].
    # #     We interpret 'subi' as the constraint index, 'subj' as which PSD block,
    # #     and (subk, subl) as the entry in that PSD block.  val is the coefficient.
    # #
    # #     In Fusion you do Xblocks[subj].index(subk, subl) to get that scalar variable.
    # #
    # for (subi, subj, subl, subk, val) in A:
    #     subi = int(subi)
    #     subj = int(subj)
    #     subl = int(subl)
    #     subk = int(subk)
    #     exprs[subi] = Expr.add(exprs[subi],
    #                            Expr.mul(val, Xblocks[subj].index(subk, subl)))
    #     if subk != subl:
    #         exprs[subi] = Expr.add(exprs[subi],
    #                            Expr.mul(val, Xblocks[subj].index(subl, subk)))

    # 3c) Finally, add these expressions as constraints = b[row].
    constraints = []
    for row in range(m):
        con = M.constraint(exprs[row], Domain.equalsTo(b[row]))
        constraints.append(con)

    #
    # 4) Objective: Maximize c^T x
    #
    print("OOO")
    M.objective(ObjectiveSense.Maximize, Expr.dot(prob_c, x))

    #
    # 5) Solver parameters
    #
    # e.g., set a limit on the number of interior-point iterations
    M.setSolverParam("intpntMaxIterations", 100)
    # M.setSolverParam("optimizerMaxTime", 1800.0)  # If you want a time limit.

    #
    # 6) Solve and extract information
    #
    try:
        M.solve()
        # Mosek Fusion may provide these solver info items:
        iter      = M.getSolverIntInfo("intpntIter")
        pfeas     = M.getSolverDoubleInfo("intpntPrimalFeas")
        dfeas     = M.getSolverDoubleInfo("intpntDualFeas")
        # Fusion does not offer a direct "get_mosek_gap()" equivalent.
        # You can query some residuals if desired:
        #   M.getSolverDoubleInfo("intpntRdinf") etc.
        max_residual = None  # or 0.0, or any other measure you wish
    except:
        # If the solver fails, you might store sentinel values
        iter         = -1
        pfeas        = -1
        dfeas        = -1
        max_residual = -1

    mosek_time = time.time() - start_time

    # Extract solution
    Xopt = []
    Sopt = []
    yopt = np.asanyarray([con.dual() for con in constraints])
    for i, dim in enumerate(s):
        data = np.array(Xblocks[i].level()).reshape(dim, dim)
        Xopt.append(data)
        data_dual = np.array(Xblocks[i].dual()).reshape(dim, dim)
        Sopt.append(data_dual)
    res = dict()
    res['Xopt'] = Xopt
    res['yopt'] = yopt
    res['Sopt'] = Sopt
    res['M'] = M
    obj_val = M.primalObjValue()

    return obj_val, res, mosek_time
