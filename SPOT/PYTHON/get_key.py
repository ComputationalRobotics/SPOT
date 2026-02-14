import numpy as np

def get_key(rpt, n):
    """
    Python equivalent of the MATLAB function:

      function key = get_key(rpt, n)
          [m, d] = size(rpt); 
          key = cell(m, 1);
          for i = 1:m
              key_tmp = 0;
              for j = 1:d
                  key_tmp = key_tmp + rpt(i, j) * (n+1)^(d-j);
              end
              key{i} = key_tmp;
          end

    Parameters
    ----------
    rpt : numpy.ndarray
        2D array of shape (m, d), where each row is an exponent pattern.
    n : int
        Base offset used in the exponent calculation.

    Returns
    -------
    keys : list
        A list of length m, where each element is an integer key.
    """
    m, d = rpt.shape
    keys = []
    for i in range(m):
        key_tmp = 0
        for j in range(d):
            # In MATLAB, exponent is (d - j), but indexing is 1-based;
            # in Python, j is 0-based, so exponent is (d - 1 - j).
            exponent = d - 1 - j
            key_tmp += int(rpt[i, j]) * (n + 1) ** exponent
        keys.append(key_tmp)  # or just key_tmp if you prefer
    return keys