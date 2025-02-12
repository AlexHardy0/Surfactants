import numpy as np
import sympy as sp

#define variables
epsilon, x = sp.symbols('epsilon x')
c = sp.Function('c')(x)

#define equation
expr = sp.diff(c,x) + c*((4+epsilon**2)*sp.sech(x)**2*(1 + (1/2)*sp.sech(x)**2))
eq = sp.Eq(sp.simplify(expr),0)
# print(eq)

#solve equation
sol = sp.solve(eq,c)
# sol = sp.simplify(sol)

print(sol)