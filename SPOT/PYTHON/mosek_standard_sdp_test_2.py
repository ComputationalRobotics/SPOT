import time
import numpy as np
import mosek
import sys


def mosek_standard_sdp_test_2(A, prob_a, b, prob_c, s):
    """
    Fully batched version using MOSEK low-level API with batch operations.

    This version eliminates Python loops by using batch APIs:
    - putvarboundslice for variable bounds
    - putaijlist for linear constraint coefficients
    - appendbarvars for all PSD variables at once
    - putbarablocktriplet for SDP constraint coefficients

    Parameters
    ----------
    A : numpy array of shape (numBarCoeff, 5)
        Each row [i, j, l, k, val] indicates that the entry (k,l)
        of the j-th PSD block is multiplied by 'val' in the i-th constraint.

    prob_a : numpy array of shape (nnzLin, 3)
        Each row [rowIdx, colIdx, val] indicates that x[colIdx] is
        multiplied by 'val' in the rowIdx-th linear constraint.

    b : 1D array-like
        Right-hand side values, length = number of constraints.

    prob_c : 1D array-like
        Objective coefficient vector c, length = number of linear variables.

    s : array-like
        Block dimensions for the PSD variables.
        Example: s = [3, 5] means two PSD blocks X0 in S^3, X1 in S^5.

    Returns
    -------
    obj_val : float
        Optimal objective value.
    res : dict
        Dictionary containing solution: {'Xopt': ..., 'yopt': ..., 'Sopt': ...}
    mosek_time : float
        Total elapsed time for problem setup and solve.
    """

    start_time = time.time()

    with mosek.Env() as env:
        with env.Task(0, 0) as task:
            # Set task name to match Fusion API output
            task.puttaskname("mosek_standard_sdp_test_2")

            # Send log output to stdout
            task.set_Stream(mosek.streamtype.log, sys.stdout.write)

            #
            # Problem dimensions
            #
            b = np.squeeze(b)
            m = len(b)
            prob_c = prob_c.astype(float)
            n = len(prob_c)
            numbarvar = len(s)

            #
            # 1) Add constraints and variables
            #
            task.appendcons(m)
            task.appendvars(n)

            #
            # 2) Set constraint bounds: A*x + <SDP terms> == b
            #
            task.putconboundslice(0, m,
                                 [mosek.boundkey.fx] * m,
                                 b.tolist(),
                                 b.tolist())

            #
            # 3) Set variable bounds: all variables are unbounded
            #
            task.putvarboundslice(0, n,
                                 [mosek.boundkey.fr] * n,
                                 [-np.inf] * n,
                                 [np.inf] * n)

            #
            # 4) Set objective: Maximize c^T x
            #
            task.putclist(list(range(n)), prob_c.tolist())
            task.putobjsense(mosek.objsense.maximize)

            #
            # 5) Add linear constraint coefficients
            #
            if len(prob_a) > 0:
                # prob_a format: [rowIdx, colIdx, val]
                row_indices = prob_a[:, 0].astype(np.int32).tolist()
                col_indices = prob_a[:, 1].astype(np.int32).tolist()
                values = prob_a[:, 2].tolist()

                # Batch operation: add all linear coefficients at once
                task.putaijlist(row_indices, col_indices, values)

            #
            # 6) Add PSD (bar) variables according to dimensions in s
            #
            s_int = [int(si) for si in s]
            task.appendbarvars(s_int)

            #
            # 7) Add SDP constraint coefficients
            #
            if len(A) > 0:
                # A format: [subi, subj, subl, subk, val]
                # subi: constraint index
                # subj: PSD block index
                # (subk, subl): entry in that PSD block
                # val: coefficient
                subi = A[:, 0].astype(np.int32).tolist()
                subj = A[:, 1].astype(np.int32).tolist()
                subl = A[:, 2].astype(np.int32).tolist()
                subk = A[:, 3].astype(np.int32).tolist()
                val = A[:, 4].astype(np.float64).tolist()

                # Batch operation: add all SDP coefficients at once
                task.putbarablocktriplet(subi, subj, subk, subl, val)

            #
            # 8) Solver parameters
            #
            task.putintparam(mosek.iparam.intpnt_max_iterations, 100)

            #
            # 9) Solve
            #
            task.optimize()

            # Print interior-point solution summary (matching Fusion API output)
            task.solutionsummary(mosek.streamtype.log)

            #
            # 10) Extract solution
            #
            yopt = np.zeros(m)
            task.gety(mosek.soltype.itr, yopt)

            Xopt = []
            Sopt = []
            for j in range(numbarvar):
                dim = int(s[j])
                lenbarvar = int(dim * (dim + 1) / 2)

                # Extract primal PSD variable
                barxj = np.zeros(lenbarvar)
                task.getbarxj(mosek.soltype.itr, j, barxj)
                Xj = np.zeros((dim, dim))
                idx = 0
                for row in range(dim):
                    for col in range(row, dim):
                        Xj[row, col] = Xj[col, row] = barxj[idx]
                        idx += 1
                Xopt.append(Xj)

                # Extract dual PSD variable
                barsj = np.zeros(lenbarvar)
                task.getbarsj(mosek.soltype.itr, j, barsj)
                Sj = np.zeros((dim, dim))
                idx = 0
                for row in range(dim):
                    for col in range(row, dim):
                        Sj[row, col] = Sj[col, row] = barsj[idx]
                        idx += 1
                Sopt.append(Sj)

            obj_val = task.getprimalobj(mosek.soltype.itr)

            res = {'Xopt': Xopt, 'yopt': yopt, 'Sopt': Sopt}
            mosek_time = time.time() - start_time

            return obj_val, res, mosek_time
