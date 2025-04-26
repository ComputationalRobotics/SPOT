import numpy as np
from SPOT.PYTHON.sorteig import sorteig

def naive_extract(Xs, mom_mat_rpt, tI, total_var_num):
    """
    Naively extracts a minimizer vector for CSTSS.
    
    Parameters:
        Xs            : List of NumPy arrays. Each element is a matrix X corresponding to a particular sub-problem.
        mom_mat_rpt   : List of 2D NumPy arrays (reports). Each report's last column holds variable indices.
        tI            : A list of the same length as mom_mat_rpt. Each element is a list of index arrays (or lists)
                        indicating which rows of the corresponding report to use.
        total_var_num : Total number of variables (an integer).
                        
    Returns:
        v_opt       : A NumPy array of length total_var_num representing the extracted minimizer (averaged over contributions).
        output_info : Dictionary containing extra output information (here, a list 'ds' of eigenvalue vectors for each submatrix).
    """
    # Initialize output vectors (using 1D arrays)
    v_opt = np.zeros(total_var_num)
    cnt = np.zeros(total_var_num)
    
    mom_mat_num = len(mom_mat_rpt)
    
    total_mat_num = 0
    for i in range(mom_mat_num):
        for ii in range(len(tI[i])):
            total_mat_num += 1
    ds = [None] * total_mat_num
    
    # Preallocate a list for eigenvalue information (one entry per submatrix)
    ds = [None] * total_mat_num

    mat_cnt = 0  # overall counter for submatrices (MATLAB 1-indexed, so we subtract 1 when indexing Xs and ds)
    for i in range(mom_mat_num):
        rpt = mom_mat_rpt[i]
        # Iterate over each index set in tI for the current report
        for ii in range(len(tI[i])):
            mat_cnt += 1
            # Extract a subset of the report rows based on the given indices.
            # Note: It is assumed that tI[i][ii] is already in 0-indexing.
            rpt_sub = rpt[tI[i][ii], :]
            
            # Determine how many rows of rpt_sub represent monomials of degree 0 or 1.
            idx = 0
            for j in range(rpt_sub.shape[0]):
                # If the row has exactly 0 or 1 nonzero elements, count it; otherwise, stop.
                if np.count_nonzero(rpt_sub[j, :]) == 1 or np.count_nonzero(rpt_sub[j, :]) == 0:
                    idx += 1
                else:
                    break
            
            # Proceed only if more than one row qualifies and the first row is a constant monomial (all zeros)
            if idx > 1 and np.count_nonzero(rpt_sub[0, :]) == 0:
                # Get the corresponding matrix from Xs. Adjust for 0-indexing.
                X = Xs[mat_cnt - 1]
                # Extract the top-left (idx x idx) submatrix.
                X_small = X[:idx, :idx]
                
                # Compute and sort the eigenvalues and eigenvectors.
                V, D = sorteig(X_small)
                # Take the first eigenvector and normalize it by its first element.
                vv = V[:, 0].copy()
                vv = vv / vv[0]
                # Discard the first element (MATLAB: vv(2:end))
                vv = vv[1:]
                
                # Get variable indices from rpt_sub rows 2 through idx (MATLAB: rpt_sub(2:idx, end)).
                # Convert to integer indices and adjust from MATLAB's 1-indexing to Python's 0-indexing.
                var_id = rpt_sub[1:idx, -1].astype(int) - 1
                
                # Accumulate the values in v_opt and update the count.
                v_opt[var_id] += vv
                cnt[var_id] += 1

                # Save the diagonal of D (the sorted eigenvalues) for this submatrix.
                ds[mat_cnt - 1] = np.diag(D)

    # Avoid division by zero for variables that have never been updated.
    cnt[cnt == 0] = 1
    v_opt = v_opt / cnt

    output_info = {"ds": ds}
    return v_opt, output_info