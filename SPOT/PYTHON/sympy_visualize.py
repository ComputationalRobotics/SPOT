import sympy as sp

def sympy_visualize(pop, x, var_str_mapping, fid):
    """
    Writes a list of sympy expressions (polynomials) into a markdown file
    with LaTeX formatting using an align environment.

    Parameters:
      pop             : list of sympy expressions (the polynomials to visualize)
      x               : list of sympy symbols used in the polynomials
      var_str_mapping : list (or tuple) of strings corresponding to each variable,
                        e.g., ["r_{c}", "r_{s}", "s_{x}", "s_{y}"] 
      fid             : file-like object (already opened for writing)

    The function builds a common monomial basis by appending the extra polynomial
      sum(x_i**2) so that every variable is included in the basis.
    """
    N = len(pop)
    # Append an extra polynomial to enforce the presence of every variable.
    extra_poly = sum([xi**2 for xi in x])
    extended_pop = pop + [extra_poly]

    # Build a global set of monomials as exponent tuples from the extended list.
    global_keys = set()
    for poly_expr in extended_pop:
        poly_obj = sp.Poly(sp.expand(poly_expr), x)
        global_keys.update(poly_obj.as_dict().keys())
    # Sort the monomials by total degree then lex order.
    global_keys = sorted(global_keys, key=lambda ex: (sum(ex), ex))
    
    pop_tex_list = []
    
    # Loop through each original polynomial.
    for p in pop:
        poly_obj = sp.Poly(sp.expand(p), x)
        # Get the monomial-to-coefficient mapping.
        poly_dict = poly_obj.as_dict()  # Keys: exponent tuples, Values: coefficients
        p_tex = ""
        first_term = True
        # Loop over the global monomial basis so the terms are in a common order.
        for ex in global_keys:
            coef = poly_dict.get(ex, 0)
            if coef == 0:
                continue  # Skip zero coefficients.
            # Decide sign. For the first term, put "-" if negative; else add " + " or " - ".
            if first_term:
                if coef < 0:
                    p_tex += "-"
                first_term = False
            else:
                if coef < 0:
                    p_tex += " - "
                else:
                    p_tex += " + "
            abs_coef = abs(coef)
            # Include the coefficient if it is not (approximately) 1 or if the term is constant.
            if (abs(abs_coef - 1) > 1e-8) or all(exp == 0 for exp in ex):
                # Use sympy's LaTeX printer for the coefficient.
                p_tex += sp.latex(abs_coef)
            # Now add variables according to their exponents.
            for i, exp in enumerate(ex):
                if exp != 0:
                    var_name = var_str_mapping[i+1]
                    if exp == 1:
                        p_tex += var_name
                    else:
                        p_tex += var_name + "^{" + str(exp) + "}"
        pop_tex_list.append(p_tex)
    
    # Now write the LaTeX code into the markdown file.
    fid.write("\n$$\n")
    fid.write("\\begin{align}\n")
    for i, line in enumerate(pop_tex_list):
        fid.write("   & " + line)
        if i < len(pop_tex_list) - 1:
            fid.write(" \\\\ \n")
        else:
            fid.write("\n")
    fid.write("\\end{align}\n")
    fid.write("$$\n")


# Example usage:
if __name__ == "__main__":
    # Define symbols.
    x, y = sp.symbols('x y')
    # Define some sample polynomials.
    pop = [2*x**2 + 0.5*x*y - 3*y**2, x - 4*y + 7]
    # Variable names for LaTeX (e.g., provided as strings).
    var_names = ["x", "y"]
    # Open a markdown file for writing.
    with open("output.md", "w") as fid:
        sympy_visualize(pop, [x, y], var_names, fid)
