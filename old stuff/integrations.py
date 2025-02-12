import sympy as sp

theta, phi, p_x, p_y, p_z  = sp.symbols('theta phi p_x, p_y, p_z')

# eq = p*sp.cos(phi)*sp.sin(theta)**2
# eq = p*sp.sin(theta)*sp.cos(theta)

eq = p_x*sp.sin(theta)*sp.cos(phi) + p_y*sp.sin(theta)*sp.sin(phi) + p_z*sp.cos(theta)

# eq = sp.sin(theta)*sp.cos(phi) + sp.sin(theta)*sp.sin(phi) + sp.cos(theta)


# eq = p_x*sp.cos(theta) + p_y*sp.sin(theta)

eq2 = sp.expand(eq**2) * sp.sin(theta)
print(eq2)

# integral = sp.integrate(eq2, (theta, 0, 2*sp.pi))
double_integral = sp.integrate(sp.integrate(eq2, (theta, 0, sp.pi)), (phi, 0, 2*sp.pi))

print(double_integral)
# print(integral)