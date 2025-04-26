import sympy as sp

def sympy_clean(pop, v, tol, if_scale):
    """
    Rescales each polynomial in pop so that the maximum absolute value 
    of its coefficient is 1 (if if_scale is True). It also removes any 
    terms with a (scaled) coefficient whose magnitude is less than tol.
    
    Parameters:
      pop      : list of sympy expressions representing the polynomials.
      v        : list of sympy symbols representing the variables.
      tol      : numerical tolerance; terms with coefficient below tol are omitted.
      if_scale : boolean flag; if True, each polynomial is scaled by its maximum absolute coefficient.
    
    Returns:
      cleaned_pop : list of cleaned (and possibly scaled) sympy expressions.
      scale_vec   : list of scale factors for each polynomial (maximum absolute coefficient if scaled, else 1).
    """
    cleaned_pop = []
    scale_vec = []
    
    # Process each polynomial in the input list
    for poly_expr in pop:
        # Expand and convert to a polynomial object in the given variables
        expanded_expr = sp.expand(poly_expr)
        poly_obj = sp.Poly(expanded_expr, v)
        poly_dict = poly_obj.as_dict()  # keys: exponent tuples, values: coefficients
        
        # Determine maximum absolute coefficient for scaling (if any term exists)
        if poly_dict:
            max_abs = max(abs(coef) for coef in poly_dict.values())
        else:
            max_abs = 0
        
        # Use the scaling factor if scaling is enabled and there is at least one nonzero coefficient.
        scale = max_abs if (if_scale and max_abs != 0) else 1
        scale_vec.append(scale)
        
        # Reconstruct the cleaned polynomial
        new_poly = 0
        for monom, coef in poly_dict.items():
            # Divide each coefficient by the scaling factor if if_scale is True.
            norm_coef = coef / scale
            if abs(norm_coef) > tol:
                term = 1
                for i, exp in enumerate(monom):
                    term *= v[i]**exp
                new_poly += norm_coef * term
        # Simplify the reconstructed expression and append to result list.
        cleaned_pop.append(sp.simplify(new_poly))
    
    return cleaned_pop, scale_vec

# Example usage:
if __name__ == '__main__':
    # Define some symbols.
    x, y, z = sp.symbols('x y z')
    # Define two example polynomials.
    pop = [2*x**2 + 0.5*x*y - 0.001*y**2, 10*x - 5*y + 0.0001, -100 * z + 0.3*x**3 ]
    # Set tolerance and scaling flag.
    tol = 1e-10
    if_scale = True
    # Clean the polynomials.
    cleaned_pop, scale_vec = sympy_clean(pop, [x, y, z], tol, if_scale)
    print("Cleaned polynomials:")
    for cp in cleaned_pop:
        sp.pprint(cp)
    print("\nScale factors:")
    print(scale_vec)
