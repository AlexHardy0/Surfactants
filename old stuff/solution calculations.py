import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# Define symbols and functions
epsilon, beta, x = sp.symbols('epsilon beta x')
phi = sp.Function('phi')(x)
phi_0 = sp.Function('phi_0')(x)
phi_1 = sp.Function('phi_1')(x)
phi_2 = sp.Function('phi_2')(x)

c = sp.Function('c')(x)
c_0 = sp.Function('c_0')(x)
c_1 = sp.Function('c_1')(x)
c_2 = sp.Function('c_2')(x)

px = sp.Function('px')(x)
px_0 = sp.Function('px_0')(x)
px_1 = sp.Function('px_1')(x)
px_2 = sp.Function('px_2')(x)

# Define the perturbative expansions for each variable
sol_phi = phi_0 + epsilon * phi_1 + epsilon**2 * phi_2
sol_c = c_0 + epsilon * c_1 + epsilon**2 * c_2
sol_px = px_0 + epsilon * px_1 + epsilon**2 * px_2

# Substitute the perturbative expansions into the original equations
eq_phi = sp.Eq(beta * (-phi + phi**3 - 0.5 * sp.diff(phi, x, x)) - epsilon * sp.diff(c * px, x), 0)
eq_c = sp.Eq(sp.diff(c, x) + c * sp.diff(px**2, x) + c * epsilon * sp.diff(px * sp.diff(phi, x), x), 0)
eq_px = sp.Eq(2 * px + epsilon * sp.diff(phi, x), 0)

eq_phi_subbed = sp.expand(eq_phi.subs({'phi': sol_phi, 'c': sol_c, 'px': sol_px}).lhs)
eq_c_subbed = sp.expand(eq_c.subs({'phi': sol_phi, 'c': sol_c, 'px': sol_px}).lhs)
eq_px_subbed = sp.expand(eq_px.subs({'phi': sol_phi, 'c': sol_c, 'px': sol_px}).lhs)

# Collect terms by powers of epsilon for each equation
eq_phi_order0 = sp.Eq(sp.collect(eq_phi_subbed, epsilon).coeff(epsilon, 0), 0)
eq_phi_order1 = sp.Eq(sp.collect(eq_phi_subbed, epsilon).coeff(epsilon, 1), 0)
eq_phi_order2 = sp.Eq(sp.collect(eq_phi_subbed, epsilon).coeff(epsilon, 2), 0)

eq_c_order0 = sp.Eq(sp.collect(eq_c_subbed, epsilon).coeff(epsilon, 0), 0)
eq_c_order1 = sp.Eq(sp.collect(eq_c_subbed, epsilon).coeff(epsilon, 1), 0)
eq_c_order2 = sp.Eq(sp.collect(eq_c_subbed, epsilon).coeff(epsilon, 2), 0)

eq_px_order0 = sp.Eq(sp.collect(eq_px_subbed, epsilon).coeff(epsilon, 0), 0)
eq_px_order1 = sp.Eq(sp.collect(eq_px_subbed, epsilon).coeff(epsilon, 1), 0)
eq_px_order2 = sp.Eq(sp.collect(eq_px_subbed, epsilon).coeff(epsilon, 2), 0)

# Solve the zeroth-order equation for phi_0, c_0, and px_0
sol_phi0 = sp.solve(eq_phi_order0, phi_0)
sol_c0 = sp.solve(eq_c_order0.subs(phi_0, sol_phi0), c_0)
sol_px0 = sp.solve(eq_px_order0.subs({phi_0: sol_phi0, c_0: sol_c0}), px_0)

# Display solutions for zeroth order
print(sol_c0)

