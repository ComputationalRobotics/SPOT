"""
NumPoly Builder: A fast polynomial construction system that completely bypasses SymPy.

This module provides:
1. NumPolyVar: A lightweight polynomial variable class (replaces SymPy symbols)
2. NumPolyExpr: A polynomial expression class supporting arithmetic operations
3. NumPolySystem: A system to manage multiple polynomials and convert to supp_rpt format

Usage:
    # Create a polynomial system with n variables
    ps = NumPolySystem(n_vars=10)

    # Get variables (0-indexed internally, but can use 1-indexed for MATLAB compatibility)
    x0, x1, x2 = ps.var(0), ps.var(1), ps.var(2)

    # Build polynomials using natural syntax
    eq1 = x0**2 + x1**2 - 1
    eq2 = x0 * x1 + 0.5 * x2

    # Add to system
    ps.add_eq(eq1)
    ps.add_eq(eq2)

    # Get supp_rpt format directly (no SymPy conversion needed!)
    supp_rpt_h, coeff_h, dj_h = ps.get_eq_supp_rpt(kappa=2)
"""

import numpy as np
from typing import List, Tuple, Dict, Union, Optional
from dataclasses import dataclass


@dataclass
class Term:
    """A single term in a polynomial: coeff * x[i1]^e1 * x[i2]^e2 * ..."""
    coeff: float
    exponents: Dict[int, int]  # var_index -> exponent

    def degree(self) -> int:
        return sum(self.exponents.values())

    def copy(self) -> 'Term':
        return Term(self.coeff, self.exponents.copy())


class NumPolyExpr:
    """
    A polynomial expression represented as a list of terms.
    Supports arithmetic operations: +, -, *, **, scalar multiplication.
    """

    def __init__(self, terms: List[Term] = None):
        self.terms = terms if terms is not None else []

    @staticmethod
    def from_var(var_index: int) -> 'NumPolyExpr':
        """Create a polynomial that is just a single variable."""
        return NumPolyExpr([Term(1.0, {var_index: 1})])

    @staticmethod
    def from_const(value: float) -> 'NumPolyExpr':
        """Create a constant polynomial."""
        if value == 0:
            return NumPolyExpr([])
        return NumPolyExpr([Term(float(value), {})])

    def copy(self) -> 'NumPolyExpr':
        return NumPolyExpr([t.copy() for t in self.terms])

    def _simplify(self) -> 'NumPolyExpr':
        """Combine like terms."""
        term_dict = {}
        for term in self.terms:
            # Convert exponents to a hashable key (sorted tuple)
            key = tuple(sorted(term.exponents.items()))
            if key in term_dict:
                term_dict[key] += term.coeff
            else:
                term_dict[key] = term.coeff

        # Build new terms list, excluding zero coefficients
        new_terms = []
        for key, coeff in term_dict.items():
            if abs(coeff) > 1e-15:
                new_terms.append(Term(coeff, dict(key)))

        return NumPolyExpr(new_terms)

    def __add__(self, other: Union['NumPolyExpr', int, float]) -> 'NumPolyExpr':
        if isinstance(other, (int, float)):
            other = NumPolyExpr.from_const(other)
        result = NumPolyExpr(self.terms + other.terms)
        return result._simplify()

    def __radd__(self, other: Union[int, float]) -> 'NumPolyExpr':
        return self.__add__(other)

    def __sub__(self, other: Union['NumPolyExpr', int, float]) -> 'NumPolyExpr':
        if isinstance(other, (int, float)):
            other = NumPolyExpr.from_const(other)
        neg_other = NumPolyExpr([Term(-t.coeff, t.exponents.copy()) for t in other.terms])
        result = NumPolyExpr(self.terms + neg_other.terms)
        return result._simplify()

    def __rsub__(self, other: Union[int, float]) -> 'NumPolyExpr':
        return NumPolyExpr.from_const(other).__sub__(self)

    def __neg__(self) -> 'NumPolyExpr':
        return NumPolyExpr([Term(-t.coeff, t.exponents.copy()) for t in self.terms])

    def __mul__(self, other: Union['NumPolyExpr', int, float]) -> 'NumPolyExpr':
        if isinstance(other, (int, float)):
            if other == 0:
                return NumPolyExpr([])
            return NumPolyExpr([Term(t.coeff * other, t.exponents.copy()) for t in self.terms])

        # Polynomial multiplication
        new_terms = []
        for t1 in self.terms:
            for t2 in other.terms:
                new_coeff = t1.coeff * t2.coeff
                new_exp = t1.exponents.copy()
                for var, exp in t2.exponents.items():
                    new_exp[var] = new_exp.get(var, 0) + exp
                new_terms.append(Term(new_coeff, new_exp))

        return NumPolyExpr(new_terms)._simplify()

    def __rmul__(self, other: Union[int, float]) -> 'NumPolyExpr':
        return self.__mul__(other)

    def __truediv__(self, other: Union[int, float]) -> 'NumPolyExpr':
        """Division by a scalar."""
        if isinstance(other, (int, float)):
            return self.__mul__(1.0 / other)
        raise TypeError("Can only divide polynomial by a scalar")

    def __pow__(self, n: int) -> 'NumPolyExpr':
        if not isinstance(n, int) or n < 0:
            raise ValueError("Exponent must be a non-negative integer")
        if n == 0:
            return NumPolyExpr.from_const(1)
        if n == 1:
            return self.copy()

        result = self.copy()
        for _ in range(n - 1):
            result = result * self
        return result

    def degree(self) -> int:
        """Return the total degree of the polynomial."""
        if not self.terms:
            return 0
        return max(t.degree() for t in self.terms)

    def to_degmat_coef(self, n_vars: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert to degmat/coef format.

        Returns:
            degmat: (n_terms, n_vars) array of exponents
            coef: (n_terms,) array of coefficients
        """
        n_terms = len(self.terms)
        if n_terms == 0:
            return np.zeros((1, n_vars), dtype=np.int32), np.array([0.0])

        degmat = np.zeros((n_terms, n_vars), dtype=np.int32)
        coef = np.zeros(n_terms, dtype=np.float64)

        for i, term in enumerate(self.terms):
            coef[i] = term.coeff
            for var_idx, exp in term.exponents.items():
                degmat[i, var_idx] = exp

        return degmat, coef

    def __repr__(self) -> str:
        if not self.terms:
            return "0"
        parts = []
        for term in self.terms:
            if not term.exponents:
                parts.append(f"{term.coeff:.4g}")
            else:
                var_parts = []
                for var, exp in sorted(term.exponents.items()):
                    if exp == 1:
                        var_parts.append(f"x{var}")
                    else:
                        var_parts.append(f"x{var}^{exp}")
                var_str = "*".join(var_parts)
                if term.coeff == 1:
                    parts.append(var_str)
                elif term.coeff == -1:
                    parts.append(f"-{var_str}")
                else:
                    parts.append(f"{term.coeff:.4g}*{var_str}")
        return " + ".join(parts).replace("+ -", "- ")


class NumPolySystem:
    """
    A system for building and managing polynomials without SymPy.

    Provides:
    - Variable creation
    - Polynomial arithmetic
    - Direct conversion to supp_rpt format for CSTSS
    """

    def __init__(self, n_vars: int):
        self.n_vars = n_vars
        self._eq_polys: List[NumPolyExpr] = []
        self._ineq_polys: List[NumPolyExpr] = []
        self._obj_poly: Optional[NumPolyExpr] = None

    def var(self, index: int) -> NumPolyExpr:
        """
        Get a polynomial variable by index (0-indexed).

        Example:
            x0 = ps.var(0)  # First variable
            x1 = ps.var(1)  # Second variable
        """
        if index < 0 or index >= self.n_vars:
            raise ValueError(f"Variable index {index} out of range [0, {self.n_vars})")
        return NumPolyExpr.from_var(index)

    def const(self, value: float) -> NumPolyExpr:
        """Create a constant polynomial."""
        return NumPolyExpr.from_const(value)

    def add_eq(self, poly: NumPolyExpr):
        """Add an equality constraint."""
        self._eq_polys.append(poly)

    def add_ineq(self, poly: NumPolyExpr):
        """Add an inequality constraint (>= 0)."""
        self._ineq_polys.append(poly)

    def set_obj(self, poly: NumPolyExpr):
        """Set the objective function."""
        self._obj_poly = poly

    def clear(self):
        """Clear all polynomials."""
        self._eq_polys = []
        self._ineq_polys = []
        self._obj_poly = None

    def _poly_to_supp_rpt(self, poly: NumPolyExpr, kappa: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert a single polynomial to supp_rpt format.

        supp_rpt format: (n_terms, 2*kappa) array where each row contains
        variable indices (1-indexed) right-aligned, sorted by degree.

        Example: x1^2 * x3 with kappa=2 -> [0, 0, 1, 1, 3] (in a 4-column array)
        """
        n_terms = len(poly.terms)
        if n_terms == 0:
            return np.zeros((1, 2 * kappa), dtype=np.float64), np.array([0.0])

        width = 2 * kappa

        # Build sequences directly from terms (faster than via degmat)
        term_data = []  # (degree, var_sequence, coef)
        for term in poly.terms:
            # Build variable sequence sorted by var index
            seq = []
            for var_idx in sorted(term.exponents.keys()):
                exp = term.exponents[var_idx]
                seq.extend([var_idx + 1] * exp)  # 1-indexed

            deg = len(seq)
            if deg > width:
                raise ValueError(f"Polynomial term degree {deg} exceeds 2*kappa={width}")

            # Sort key: (degree, -var_indices for descending order)
            sort_key = (deg, tuple(-v for v in seq) if seq else ())
            term_data.append((sort_key, seq, term.coeff))

        # Sort by degree, then by variable indices
        term_data.sort(key=lambda x: x[0])

        # Build output arrays
        seqs = np.zeros((n_terms, width), dtype=np.float64)
        coef = np.zeros(n_terms, dtype=np.float64)

        for i, (_, seq, c) in enumerate(term_data):
            coef[i] = c
            if seq:
                seqs[i, -len(seq):] = seq

        return seqs, coef

    def _clean_poly(self, poly: NumPolyExpr, tol: float, if_scale: bool) -> Tuple[NumPolyExpr, float]:
        """
        Clean a polynomial: scale and remove small coefficients.

        Returns:
            cleaned polynomial, scale factor
        """
        if not poly.terms:
            return poly, 1.0

        # Find max absolute coefficient
        max_abs = max(abs(t.coeff) for t in poly.terms)
        if max_abs == 0:
            return NumPolyExpr([]), 1.0

        scale = max_abs if if_scale else 1.0

        # Scale and filter
        new_terms = []
        for t in poly.terms:
            new_coeff = t.coeff / scale
            if abs(new_coeff) > tol:
                new_terms.append(Term(new_coeff, t.exponents.copy()))

        return NumPolyExpr(new_terms), scale

    def clean_all(self, tol: float = 1e-14, if_scale: bool = True, scale_obj: bool = False):
        """Clean all polynomials in the system.

        Args:
            tol: Tolerance for removing small coefficients.
            if_scale: Whether to scale constraint polynomials.
            scale_obj: Whether to scale the objective function. Default False
                       since scaling changes the optimal value.
        """
        cleaned_eq = []
        for poly in self._eq_polys:
            cleaned, _ = self._clean_poly(poly, tol, if_scale)
            cleaned_eq.append(cleaned)
        self._eq_polys = cleaned_eq

        cleaned_ineq = []
        for poly in self._ineq_polys:
            cleaned, _ = self._clean_poly(poly, tol, if_scale)
            cleaned_ineq.append(cleaned)
        self._ineq_polys = cleaned_ineq

        if self._obj_poly:
            self._obj_poly, _ = self._clean_poly(self._obj_poly, tol, scale_obj)

    def get_supp_rpt_data(self, kappa: int) -> dict:
        """
        Get all polynomial data in supp_rpt format for CSTSS.

        Args:
            kappa: Relaxation order

        Returns:
            Dictionary with supp_rpt_f, coeff_f, supp_rpt_g, coeff_g,
            supp_rpt_h, coeff_h, dj_g, dj_h
        """
        d = 2 * kappa

        # Objective
        if self._obj_poly:
            supp_rpt_f, coeff_f = self._poly_to_supp_rpt(self._obj_poly, kappa)
        else:
            supp_rpt_f = np.zeros((1, d), dtype=np.float64)
            coeff_f = np.array([0.0])

        # Inequality constraints
        m_ineq = len(self._ineq_polys)
        supp_rpt_g = [None] * m_ineq
        coeff_g = [None] * m_ineq
        dj_g = np.zeros(m_ineq)

        for i, poly in enumerate(self._ineq_polys):
            supp_rpt_g[i], coeff_g[i] = self._poly_to_supp_rpt(poly, kappa)
            dj_g[i] = np.ceil(poly.degree() / 2)

        # Equality constraints
        m_eq = len(self._eq_polys)
        supp_rpt_h = [None] * m_eq
        coeff_h = [None] * m_eq
        dj_h = np.zeros(m_eq)

        for i, poly in enumerate(self._eq_polys):
            supp_rpt_h[i], coeff_h[i] = self._poly_to_supp_rpt(poly, kappa)
            dj_h[i] = poly.degree()

        return {
            'supp_rpt_f': supp_rpt_f,
            'coeff_f': coeff_f,
            'supp_rpt_g': supp_rpt_g,
            'coeff_g': coeff_g,
            'supp_rpt_h': supp_rpt_h,
            'coeff_h': coeff_h,
            'dj_g': dj_g,
            'dj_h': dj_h,
        }



def numpoly_visualize(supp_rpt_list, coeff_list, var_mapping, fid):
    """
    Write polynomials in supp_rpt/coeff format to a markdown file with LaTeX.
    Works directly on supp_rpt/coeff data without sympy conversion.

    supp_rpt format: each row is a variable index sequence (1-indexed, right-aligned,
    0-padded on the left). E.g. x2^2 * x5 with kappa=2 -> [0, 2, 2, 5].

    Parameters:
      supp_rpt_list : list of 2D numpy arrays (each: n_terms x 2*kappa)
      coeff_list    : list of 1D numpy arrays (each: n_terms, coefficients)
      var_mapping   : dict mapping 1-indexed variable id to LaTeX string
      fid           : file-like object (already opened for writing)
    """
    pop_tex_list = []
    for supp_rpt, coeff in zip(supp_rpt_list, coeff_list):
        p_tex = ""
        first_term = True
        for i in range(len(coeff)):
            c = coeff[i]
            if abs(c) < 1e-15:
                continue
            # Convert index sequence to (var_id, exponent) pairs
            # e.g. [0, 2, 2, 5] -> {2: 2, 5: 1}
            var_exp = {}
            for idx in supp_rpt[i]:
                v_id = int(idx)
                if v_id > 0:
                    var_exp[v_id] = var_exp.get(v_id, 0) + 1
            # Sign handling
            if first_term:
                if c < 0:
                    p_tex += "-"
                first_term = False
            else:
                if c < 0:
                    p_tex += " - "
                else:
                    p_tex += " + "
            abs_c = abs(c)
            # Print coefficient if not ~1 or if constant term
            if abs(abs_c - 1.0) > 1e-8 or len(var_exp) == 0:
                p_tex += f"{abs_c:g}"
            # Print variables sorted by id
            for v_id in sorted(var_exp.keys()):
                exp = var_exp[v_id]
                var_name = var_mapping.get(v_id, f"v_{{{v_id}}}")
                if exp == 1:
                    p_tex += var_name
                else:
                    p_tex += var_name + "^{" + str(exp) + "}"
        pop_tex_list.append(p_tex)

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


# =============================================================================
# Test and benchmark
# =============================================================================

if __name__ == '__main__':
    import time

    print("=" * 60)
    print("NumPolyBuilder Test")
    print("=" * 60)

    # Test basic operations
    ps = NumPolySystem(n_vars=5)
    x0, x1, x2, x3, x4 = [ps.var(i) for i in range(5)]

    # Build some polynomials
    eq1 = x0**2 + x1**2 - 1
    eq2 = x0 * x1 + 0.5 * x2 - 0.001 * x3
    eq3 = 2.0 * x0**2 - 3.0 * x1 * x2 + x4

    print("\nTest polynomials:")
    print(f"  eq1 = {eq1}")
    print(f"  eq2 = {eq2}")
    print(f"  eq3 = {eq3}")

    # Test degree
    print(f"\nDegrees: eq1={eq1.degree()}, eq2={eq2.degree()}, eq3={eq3.degree()}")

    # Test supp_rpt conversion
    ps.add_eq(eq1)
    ps.add_eq(eq2)
    ps.add_eq(eq3)
    ps.set_obj(x0**2 + x1**2)

    data = ps.get_supp_rpt_data(kappa=2)
    print("\nsupp_rpt_h[0] (eq1):")
    print(data['supp_rpt_h'][0])
    print("coeff_h[0]:", data['coeff_h'][0])

    # Benchmark: simulate pushT scale (N=15)
    print("\n" + "=" * 60)
    print("Benchmark: Simulating pushT scale (N=15)")
    print("=" * 60)

    n_vars = 274  # 18*15 + 4
    n_eq = 250
    n_ineq = 200

    ps2 = NumPolySystem(n_vars=n_vars)

    # Build random polynomials
    np.random.seed(42)

    print("\nBuilding polynomials...")
    start = time.time()

    def make_random_poly(ps, n_vars, max_degree=2):
        """Generate a random polynomial with total degree <= max_degree."""
        poly = ps.const(0)
        n_terms = np.random.randint(5, 11)
        for _ in range(n_terms):
            coef = np.random.randn()
            # Choose term type: degree 1 or degree 2
            deg = np.random.randint(1, max_degree + 1)
            if deg == 1:
                vi = np.random.randint(0, n_vars)
                term = coef * ps.var(vi)
            else:  # deg == 2
                choice = np.random.randint(0, 2)
                if choice == 0:
                    # x_i^2
                    vi = np.random.randint(0, n_vars)
                    term = coef * (ps.var(vi) ** 2)
                else:
                    # x_i * x_j
                    vi, vj = np.random.choice(n_vars, size=2, replace=False)
                    term = coef * ps.var(vi) * ps.var(vj)
            poly = poly + term
        return poly

    eq_polys = []
    for _ in range(n_eq):
        eq_polys.append(make_random_poly(ps2, n_vars, max_degree=2))

    ineq_polys = []
    for _ in range(n_ineq):
        ineq_polys.append(make_random_poly(ps2, n_vars, max_degree=2))

    build_time = time.time() - start
    print(f"Build time: {build_time:.4f}s")

    # Add to system
    for poly in eq_polys:
        ps2.add_eq(poly)
    for poly in ineq_polys:
        ps2.add_ineq(poly)
    ps2.set_obj(ps2.var(0)**2 + ps2.var(1)**2)

    # Clean
    print("\nCleaning polynomials...")
    start = time.time()
    ps2.clean_all(tol=1e-14, if_scale=True)
    clean_time = time.time() - start
    print(f"Clean time: {clean_time:.6f}s")

    # Convert to supp_rpt
    print("\nConverting to supp_rpt format...")
    start = time.time()
    data = ps2.get_supp_rpt_data(kappa=2)
    convert_time = time.time() - start
    print(f"Convert time: {convert_time:.4f}s")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Build polynomials: {build_time:.4f}s")
    print(f"  Clean polynomials: {clean_time:.6f}s")
    print(f"  Convert to supp_rpt: {convert_time:.4f}s")
    print(f"  Total: {build_time + clean_time + convert_time:.4f}s")
    print("=" * 60)

    # -----------------------------------------------------------------
    # Example 1: Basic polynomial arithmetic (like sympy_clean example)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 1: Basic Polynomial Arithmetic")
    print("=" * 60)

    ps_ex = NumPolySystem(n_vars=3)
    x0, x1, x2 = ps_ex.var(0), ps_ex.var(1), ps_ex.var(2)

    p1 = 2 * x0**2 + 0.5 * x0 * x1 - 0.001 * x1**2
    p2 = 10 * x0 - 5 * x1 + 0.0001
    p3 = -100 * x2 + 0.3 * x0**3

    print(f"  p1 = {p1}")
    print(f"  p2 = {p2}")
    print(f"  p3 = {p3}")
    print(f"  p1 + p2 = {p1 + p2}")
    print(f"  p1 * p2 = {p1 * p2}")
    print(f"  -p3 = {-p3}")
    print(f"  p1 degree = {p1.degree()}")

    # -----------------------------------------------------------------
    # Example 2: Clean polynomials (like sympy_clean example)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 2: Clean Polynomials (scale + remove small terms)")
    print("=" * 60)

    ps_clean = NumPolySystem(n_vars=3)
    x0, x1, x2 = ps_clean.var(0), ps_clean.var(1), ps_clean.var(2)

    ps_clean.add_eq(2 * x0**2 + 0.5 * x0 * x1 - 0.001 * x1**2)
    ps_clean.add_eq(10 * x0 - 5 * x1 + 0.0001)
    ps_clean.set_obj(-100 * x2 + 0.3 * x0**3)

    print("Before cleaning:")
    for i, p in enumerate(ps_clean._eq_polys):
        print(f"  eq[{i}] = {p}")
    print(f"  obj   = {ps_clean._obj_poly}")

    ps_clean.clean_all(tol=1e-3, if_scale=True)

    print("After cleaning (tol=1e-3, scale=True):")
    for i, p in enumerate(ps_clean._eq_polys):
        print(f"  eq[{i}] = {p}")
    print(f"  obj   = {ps_clean._obj_poly}")

    # -----------------------------------------------------------------
    # Example 3: Convert to supp_rpt format for CSTSS
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 3: supp_rpt format conversion")
    print("=" * 60)

    ps_sr = NumPolySystem(n_vars=3)
    x0, x1, x2 = ps_sr.var(0), ps_sr.var(1), ps_sr.var(2)

    ps_sr.add_eq(x0**2 + x1**2 - 1)
    ps_sr.add_ineq(x0 + x1 + x2)
    ps_sr.set_obj(x0**2 + x1**2)

    data = ps_sr.get_supp_rpt_data(kappa=2)
    print("Equality constraint  x0^2 + x1^2 - 1:")
    print(f"  supp_rpt = \n{data['supp_rpt_h'][0]}")
    print(f"  coeff    = {data['coeff_h'][0]}")
    print(f"  dj_h     = {data['dj_h']}")
    print("Inequality constraint  x0 + x1 + x2 >= 0:")
    print(f"  supp_rpt = \n{data['supp_rpt_g'][0]}")
    print(f"  coeff    = {data['coeff_g'][0]}")
    print(f"  dj_g     = {data['dj_g']}")

    # -----------------------------------------------------------------
    # Example 4: numpoly_visualize (like sympy_visualize example)
    # -----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Example 4: numpoly_visualize (write LaTeX to markdown)")
    print("=" * 60)

    var_mapping = {1: "x", 2: "y", 3: "z"}
    supp_rpt_list = [data['supp_rpt_h'][0], data['supp_rpt_g'][0]]
    coeff_list = [data['coeff_h'][0], data['coeff_g'][0]]

    with open("numpoly_output.md", "w") as fid:
        fid.write("# NumPoly Visualize Example\n")
        numpoly_visualize(supp_rpt_list, coeff_list, var_mapping, fid)

    print("  Written to numpoly_output.md")
    with open("numpoly_output.md", "r") as fid:
        print(fid.read())
