import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

# epsilon, beta, x, c0, a = sp.symbols('epsilon beta x c0 a')
# phi = sp.Function('phi')(x)


# phi = sp.tanh(x) + (0.5*(epsilon**2)*c0*x*(sp.sech(x))**2 / beta)

# dphi_dx = sp.diff(phi,x)

# print(dphi_dx)
dx = 0.5
Lx= 512 * dx
lx = (Lx-2*dx)/2
x = np.arange(-lx+dx,lx+dx,0.01)

#sech function
def sech(x):
    return 1/np.cosh(x)

c0 = 0.015
ep = 0.1

c_pert = c0*(1+0.25*ep**2*sech(x)**4)
print(np.max(c_pert))

epsilon = 1.0
W = 1.0

psi_c = np.exp(-0.5*epsilon - 0.5*W)

psi_0 = c0/(c0+psi_c)
print(psi_0)

# fig,ax = plt.subplots()

# ax.plot(x,c_pert)

# plt.show()