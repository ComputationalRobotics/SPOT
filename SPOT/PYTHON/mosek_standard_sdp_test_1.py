import time
import numpy as np
import mosek
import sys


def mosek_standard_sdp_test_1(A, C, b, s):
    """

    Solves a pure SDP (no linear variables):
        min  sum_j <C_j, X_j>
        s.t. sum_j <A_ij, X_j> = b_i,   i = 0, ..., m-1
             X_j >= 0 (PSD)

    This version eliminates Python loops by using batch APIs:
    - appendbarvars for all PSD variables at once
    - putbarcblocktriplet for objective SDP coefficients
    - putbarablocktriplet for constraint SDP coefficients
    - putconboundslice for constraint bounds

    Parameters
    ----------
    A : numpy array of shape (numBarCoeff, 5)
        Each row [i, j, l, k, val] indicates that the entry (k,l)
        of the j-th PSD block has coefficient 'val' in the i-th constraint.

    C : numpy array of shape (numObjCoeff, 4)
        Each row [block_idx, col, row_idx, val] indicates that the entry
        (row_idx, col) of the block_idx-th PSD block has coefficient 'val'
        in the objective function.

    b : 1D array-like
        Right-hand side values, length = number of constraints.

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
            task.puttaskname("MySDP")

            # Send log output to stdout
            task.set_Stream(mosek.streamtype.log, sys.stdout.write)

            #
            # Problem dimensions
            #
            b = np.squeeze(b)
            m = len(b)
            numbarvar = len(s)

            #
            # 1) Add constraints (no linear variables in pure SDP)
            #
            task.appendcons(m)

            #
            # 2) Set constraint bounds: <SDP terms> == b
            #
            task.putconboundslice(0, m,
                                 [mosek.boundkey.fx] * m,
                                 b.tolist(),
                                 b.tolist())

            #
            # 3) Add PSD (bar) variables according to dimensions in s
            #
            s_int = [int(si) for si in s]
            task.appendbarvars(s_int)

            #
            # 4) Set objective: Minimize sum_j <C_j, X_j>
            #
            task.putobjsense(mosek.objsense.minimize)

            if len(C) > 0:
                # C format: [block_idx, col, row_idx, val]
                obj_subj = C[:, 0].astype(np.int32).tolist()
                obj_subl = C[:, 1].astype(np.int32).tolist()
                obj_subk = C[:, 2].astype(np.int32).tolist()
                obj_val_list = C[:, 3].astype(np.float64).tolist()

                # Batch operation: set all objective SDP coefficients at once
                task.putbarcblocktriplet(obj_subj, obj_subk, obj_subl, obj_val_list)

            #
            # 5) Add SDP constraint coefficients
            #
            if len(A) > 0:
                # A format: [subi, subj, subl, subk, val]
                subi = A[:, 0].astype(np.int32).tolist()
                subj = A[:, 1].astype(np.int32).tolist()
                subl = A[:, 2].astype(np.int32).tolist()
                subk = A[:, 3].astype(np.int32).tolist()
                val = A[:, 4].astype(np.float64).tolist()

                # Batch operation: add all SDP constraint coefficients at once
                task.putbarablocktriplet(subi, subj, subk, subl, val)

            #
            # 6) Solver parameters
            #
            task.putintparam(mosek.iparam.intpnt_max_iterations, 100)

            #
            # 7) Solve
            #
            task.optimize()

            # Print interior-point solution summary (matching Fusion API output)
            task.solutionsummary(mosek.streamtype.log)

            #
            # 8) Extract solution
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
