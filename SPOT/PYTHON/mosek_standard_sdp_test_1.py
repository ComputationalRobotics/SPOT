from collections import defaultdict
import time
import numpy as np
import mosek
import sys
from mosek.fusion import Model, Domain, Expr, ObjectiveSense

def reconstruct_sym_matrix(flat, n):
    """
    Given a flattened lower-triangular array 'flat' corresponding to an n-by-n symmetric matrix,
    reconstruct the full symmetric matrix.
    """
    X_full = np.zeros((n, n))
    idx = 0
    for i in range(n):
        for j in range(i+1):
            X_full[i, j] = flat[idx]
            X_full[j, i] = flat[idx]
            idx += 1
    return X_full

# ------------------------------------------------------------------
# helper: linear index of a lower-triangular entry (k ≥ l)
# MOSEK packs (k, l)  →  k(k+1)/2 + l      with 0-based indices.
# ------------------------------------------------------------------
def pack_tril(k: int, l: int) -> int:
    return k * (k + 1) // 2 + l

def mosek_standard_sdp_test_1(A, C, b, s):
    """
    Solve an SDP in Python using the Fusion API.
    This replaces the old dictionary-based 'mosekopt' approach.
    """
    start_time = time.time()

    with Model("MySDP") as M:
        # (Optional) Turn on detailed logging
        M.setSolverParam("log", 1)
        
        # Send log output to stdout
        M.setLogHandler(sys.stdout)

        # Create PSD block variables
        Xvars = []
        for i, dim in enumerate(s):
            Xvars.append(M.variable(f"X_{i}", [dim, dim], Domain.inPSDCone()))

        # Build the objective from C
        obj_expr = Expr.constTerm(0.0)
        for row in C:
            block_idx = int(row[0])
            col       = int(row[1])
            row_idx   = int(row[2])
            val       = row[3]
            obj_expr  = Expr.add(obj_expr, Expr.mul(val, Xvars[block_idx].index(row_idx, col)))
            if row_idx != col:
                obj_expr  = Expr.add(obj_expr, Expr.mul(val, Xvars[block_idx].index(col, row_idx)))

        M.objective("obj", ObjectiveSense.Minimize, obj_expr)

        # Build the constraints from A
        constraints = []
        for i_con, rhs_val in enumerate(b):
            con_expr = Expr.constTerm(0.0)
            # Gather rows in A with A[row,0] == i_con
            matched_rows = np.where(A[:,0] == i_con)[0]
            for r_idx in matched_rows:
                block_idx = int(A[r_idx, 1])
                col       = int(A[r_idx, 2])
                row_idx   = int(A[r_idx, 3])
                val       = A[r_idx, 4]
                con_expr  = Expr.add(con_expr, Expr.mul(val, Xvars[block_idx].index(row_idx, col)))
                if row_idx != col:
                    con_expr  = Expr.add(con_expr, Expr.mul(val, Xvars[block_idx].index(col, row_idx)))

            con = M.constraint(f"con_{i_con}", con_expr, Domain.equalsTo(rhs_val))
            constraints.append(con)

        # Solve
        M.setSolverParam("intpntMaxIterations", 100)
        M.solve()
        solve_time = time.time() - start_time

        # Extract solution
        Xopt = []
        Sopt = []
        yopt = np.asanyarray([con.dual() for con in constraints])
        for i, dim in enumerate(s):
            data = np.array(Xvars[i].level()).reshape(dim, dim)
            Xopt.append(data)
            data_dual = np.array(Xvars[i].dual()).reshape(dim, dim)
            Sopt.append(data_dual)
        res = dict()
        res['Xopt'] = Xopt
        res['yopt'] = yopt
        res['Sopt'] = Sopt
        res['M'] = M
        obj_val = M.primalObjValue()

    return obj_val, res, solve_time



