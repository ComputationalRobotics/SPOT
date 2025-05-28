import numpy as np
from math import comb  

from SPOT.PYTHON.get_key import get_key 
from SPOT.PYTHON.sorteig import sorteig

# -----------------------------------------------------------------------------
def extraction_robust(input_info):
    """
    Python equivalent of:
       function [xi_recover, output_info] = extraction_robust(input_info)

    input_info should have:
        - input_info["mom_sub"] : sub-moment matrix
        - input_info["Ks"]      : list of localizing matrices
        - input_info["eps"]     : truncated tolerance
    Returns:
        - xi_recover : shape (n, rank_of_mom_sub)
        - output_info: dictionary with w_recover and v_recover
    """
    Ks = input_info["Ks"]
    mom_sub = input_info["mom_sub"]
    eps_val = input_info["eps"]

    n = len(Ks)

    # 1) Diagonalize mom_sub (sorted)
    U, S_mat = sorteig(mom_sub)
    s = np.diag(S_mat)

    # 2) Truncate small eigenvalues
    if_truncate = False
    k_trunc = len(s)
    # s[0] is the largest eigenvalue if sorted descending
    for k in range(len(s) - 1):
        if s[k + 1] / s[0] < eps_val:
            if_truncate = True
            k_trunc = k + 1  # Keep eigenvalues up to index k
            break

    if if_truncate:
        S_mat = S_mat[:k_trunc, :k_trunc]
        U = U[:, :k_trunc]
        s = s[:k_trunc]

    # 3) Compute S_sqrt and S_sqrt_inv
    S_sqrt = np.diag(np.sqrt(s))
    S_sqrt_inv = np.diag(1.0 / np.sqrt(s))

    # 4) Build YK = S_sqrt_inv * U' * Ks[i] * U * S_sqrt_inv
    YKs = []
    U_trans = U.T
    for K in Ks:
        YK = S_sqrt_inv @ (U_trans @ (K @ (U @ S_sqrt_inv)))
        # Ensure numerical symmetry
        # (Optional depending on your usage; sometimes small numeric imbalances occur)
        # YK = 0.5 * (YK + YK.T)
        YKs.append(YK)

    # 5) Construct a random linear combination YK_random
    #    to diagonalize and get an "O" matrix
    YK_random = np.zeros_like(YKs[0])
    for i in range(n):
        rand_num = 2.0 * np.random.rand() - 1.0  # random in [-1, 1]
        YK_random += YKs[i] * rand_num
    YK_random = 0.5 * (YK_random + YK_random.T)

    # Diagonalize YK_random
    eigvals_rand, O = np.linalg.eig(YK_random)
    # We don't strictly need sorted eigenvectors here, but we could sort them if desired.

    # 6) For each Ks[i], build Y = O' * YKs[i] * O, and store diag(Y)
    xi_recover = np.zeros((n, k_trunc))
    O_trans = O.T
    for i in range(n):
        Y = O_trans @ (YKs[i] @ O)
        # diag(Y) becomes a row in xi_recover
        xi_recover[i, :] = np.diag(Y)

    # 7) Construct v_recover = abs(tmp(:,1)) in MATLAB
    tmp = O_trans @ (S_sqrt @ U_trans)
    v_recover = np.abs(tmp[:, 0])
    w_recover = v_recover**2

    output_info = {
        "w_recover": w_recover,
        "v_recover": v_recover
    }
    return xi_recover, output_info

def generate_Ks(X, rpt, total_var_num):
    """
    Minimal-change, MATLAB-faithful rewrite of your old version.
    """
    s      = rpt.shape[0]
    kappa  = rpt.shape[1] // 2
    n_total = total_var_num

    # MATLAB: if nnz(rpt(i,:)) == 1 
    n_clique = 0
    nonzero_counts = np.count_nonzero(rpt, axis=1)
    for count in nonzero_counts:
        if count == 1:
            n_clique += 1

    # ------------------------------------------------------------------
    # Build M1, M2 exactly as MATLAB does:
    #   M1 = repmat(rpt_short, s, 1)
    #   M2 = kron(rpt_short, ones(s,1))
    # ------------------------------------------------------------------
    rpt_short = rpt[:, kappa:]                      # second half of columns
    M1 = np.tile   (rpt_short, (s, 1))              # ### CHANGE ###
    M2 = np.repeat (rpt_short,  s, axis=0)          # ### CHANGE ###
    Mv = np.sort(np.hstack([M1, M2]), axis=1)

    # Dictionary  key → X(i,j)
    keys = get_key(Mv, n_total)
    C    = {}
    idx  = 0
    for i in range(s):
        for j in range(s):
            C[keys[idx]] = X[i, j]
            idx += 1

    # ------------------------------------------------------------------
    # Sub-matrix size: nchoosek(n+kappa-1, kappa-1) in MATLAB
    # ------------------------------------------------------------------
    sub_size = comb(n_clique + kappa - 1, kappa - 1)       # ### CHANGE ###

    mom_sub        = X[:sub_size, :sub_size]
    rpt_sub_short  = rpt_short[:sub_size, :]

    M1_sub = np.tile   (rpt_sub_short, (sub_size, 1))
    M2_sub = np.repeat (rpt_sub_short,  sub_size, axis=0)
    Mv_sub = np.sort(np.hstack([M1_sub, M2_sub]), axis=1)
    Mv_sub_short = Mv_sub                           # ### CHANGE ###

    # ------------------------------------------------------------------
    # Localising matrices: MATLAB builds *n* of them
    # ------------------------------------------------------------------
    Ks = []
    for i in range(n_clique):                              # ### CHANGE ###
        base_row = rpt_sub_short[i + 1, :]          # row i+1 in MATLAB
        rpt_short_single = np.repeat(base_row[np.newaxis, :],
                                     sub_size**2, axis=0)

        Kv      = np.sort(np.hstack([rpt_short_single, Mv_sub_short]), axis=1)
        id_list = get_key(Kv, n_total)

        K = np.zeros_like(mom_sub)
        idx_k = 0
        for j1 in range(sub_size):
            for j2 in range(sub_size):
                K[j1, j2] = C[id_list[idx_k]]
                idx_k += 1
        Ks.append(K)

    # ------------------------------------------------------------------
    # Variable indices: rows 2 … n+1 in MATLAB, last column
    # ------------------------------------------------------------------
    var_id = rpt[1 : n_clique + 1, -1].astype(int)         # ### CHANGE ###

    return mom_sub, Ks, var_id


def robust_extract_CS(Xs, mom_mat_rpt, total_var_num, eps_val):
    """
    Equivalent to the MATLAB:
      function [sol, output_info] = robust_extract_CS(Xs, mom_mat_rpt, total_var_num, eps)

    Args:
        Xs           : list of moment matrices (at least as long as mom_mat_rpt)
        mom_mat_rpt  : list of exponent-pattern arrays (one per clique)
        total_var_num: integer number of polynomial variables
        eps_val      : tolerance for truncation in extraction

    Returns:
        sol         : 1D NumPy array (size total_var_num)
        output_info : dict with sub-information:
            - xi_recovers
            - w_recovers
            - var_ids
    """
    mom_mat_num = len(mom_mat_rpt)

    # Initialize solution and count
    sol = np.zeros(total_var_num, dtype=float)
    cnt = np.zeros(total_var_num, dtype=float)

    # Prepare lists to store intermediate data
    xi_recovers = [None] * mom_mat_num
    w_recovers = [None] * mom_mat_num
    var_ids = [None] * mom_mat_num

    # Main loop over each moment-matrix block
    for i in range(mom_mat_num):
        rpt = mom_mat_rpt[i]    # 2D array of exponents for block i
        X = Xs[i]               # The moment matrix for block i

        # 1) Generate sub-matrix and localizing matrices
        mom_sub, Ks, var_id = generate_Ks(X, rpt, total_var_num)

        # 2) Call extraction_robust
        input_info = {
            "mom_sub": mom_sub,
            "Ks": Ks,
            "eps": eps_val
        }
        xi_recover, output_info_inner = extraction_robust(input_info)

        # 3) Identify the largest weight component in w_recover
        max_id = np.argmax(output_info_inner["w_recover"])

        # 4) Accumulate into sol and increment count
        sol[var_id - 1] += xi_recover[:, max_id]
        cnt[var_id - 1] += 1

        # Store intermediate info
        xi_recovers[i] = xi_recover
        w_recovers[i] = output_info_inner["w_recover"]
        var_ids[i] = var_id

    # Avoid divide-by-zero
    cnt[cnt == 0] = 1
    sol /= cnt

    # Build output_info
    output_info = {
        "xi_recovers": xi_recovers,
        "w_recovers": w_recovers,
        "var_ids": var_ids
    }

    return sol, output_info

def ordered_extract_CS(Xs, mom_mat_rpt, total_var_num, eps_val, cliques_rank):
    """
    Python equivalent of the MATLAB function:

        function [sol, output_info] = ordered_extract_CS( ...)
    
    Parameters
    ----------
    Xs : list of numpy.ndarray
        List of moment matrices (each a 2D array). Length >= len(mom_mat_rpt).
    mom_mat_rpt : list of numpy.ndarray
        Each entry is a 2D array of exponents corresponding to one "side" of the
        moment matrix structure.
    total_var_num : int
        Number of polynomial variables.
    eps_val : float
        Truncation tolerance for eigenvalue-based extraction.
    cliques_rank : list or numpy.ndarray of int
        A permutation or ranking of the cliques, indicating the order in which
        they should be processed/extracted.
    
    Returns
    -------
    sol : numpy.ndarray
        1D array of size (total_var_num,), containing the extracted solution.
    output_info : dict
        A dictionary with the following keys:
            - "xi_recovers": list of arrays (the robust-extracted xi for each clique)
            - "w_recovers" : list of 1D arrays of weights from `extraction_robust`
            - "var_ids"    : list of arrays, each containing the variable indices
                             for the corresponding clique
            - "cnt"        : 1D array of counts (how many times each variable was updated)
    """
    mom_mat_num = len(mom_mat_rpt)

    # Initialize solution and counts
    sol = np.zeros(total_var_num, dtype=float)
    cnt = np.zeros(total_var_num, dtype=float)

    # Prepare lists to store intermediate extraction results
    xi_recovers = [None] * mom_mat_num
    w_recovers = [None] * mom_mat_num
    var_ids = [None] * mom_mat_num

    # Iterate over cliques in the specified order
    # Note: MATLAB indexing is 1-based; Python is 0-based. So:
    #  for i in 1:mom_mat_num --> for i in range(mom_mat_num)
    for i in range(mom_mat_num):
        # 'ii' is the actual index of the clique we want to process in this iteration
        ii = cliques_rank[i]

        # Fetch the exponent pattern (rpt) and corresponding moment matrix (X)
        rpt = mom_mat_rpt[ii]
        X = Xs[ii]

        # 1) Build sub-matrix and localizing matrices
        mom_sub, Ks, var_id = generate_Ks(X, rpt, total_var_num)

        # 2) Extract robust solution from each localizing block
        input_info = {
            "mom_sub": mom_sub,
            "Ks": Ks,
            "eps": eps_val
        }
        xi_recover, output_info_inner = extraction_robust(input_info)

        # 3) Determine which "column" of xi_recover to use
        #    (xi_recover has shape [nVarsInBlock, rank_of_mom_sub]).
        #    We either pick the column with largest w_recover or the one that
        #    best fits the current solution (smallest deviation).
        # MATLAB: if i == 1 || length(output_info_inner.w_recover) == 1 -> pick max
        # In Python, i == 0 is the first iteration. We do the same logic:
        if i == 0 or len(output_info_inner["w_recover"]) == 1:
            # Pick the maximum weight entry
            max_id = np.argmax(output_info_inner["w_recover"])
            for j in range(xi_recover.shape[0]):
                idx = var_id[j] - 1  # The global variable index
                # Weighted update of sol[idx]
                sol[idx] = (sol[idx] * cnt[idx] + xi_recover[j, max_id]) / (1.0 + cnt[idx])
                cnt[idx] += 1.0

        else:
            # Compute the deviation from the current sol for each possible column of xi_recover
            num_candidates = xi_recover.shape[1]  # rank_of_mom_sub
            deviate = np.zeros(num_candidates, dtype=float)

            # For each column j, sum (xi_recover[k, j] - sol[idx])^2 for variables with cnt>0
            for j in range(num_candidates):
                tmp = 0.0
                for k in range(xi_recover.shape[0]):
                    idx = var_id[k] - 1
                    if cnt[idx] > 0:
                        diff = xi_recover[k, j] - sol[idx]
                        tmp += diff * diff
                deviate[j] = tmp

            # Pick the column with the smallest total deviation
            min_id = np.argmin(deviate)
            for j in range(xi_recover.shape[0]):
                idx = var_id[j] - 1
                sol[idx] = (sol[idx] * cnt[idx] + xi_recover[j, min_id]) / (1.0 + cnt[idx])
                cnt[idx] += 1.0

        # 4) Store intermediate results in the corresponding position
        xi_recovers[ii] = xi_recover
        w_recovers[ii] = output_info_inner["w_recover"]
        var_ids[ii] = var_id

    # Build the output_info dictionary
    output_info = {
        "xi_recovers": xi_recovers,
        "w_recovers": w_recovers,
        "var_ids": var_ids,
        "cnt": cnt
    }

    return sol, output_info

