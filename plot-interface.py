import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

Nx, Ny = 256,8
dx = 0.25
Lx, Ly = Nx*dx, Ny*dx
n = int(9e6)
xshift = 0.125

# sech function
def sech(x):
    return 1/np.cosh(x)

phi_load = np.loadtxt(f'./interface-surfactant-5-np-400-data/phi{n}.txt')

sl = int(Ny/2)
x = np.arange(-Lx/2+xshift, Lx/2+xshift, dx)
x_a = np.arange(-Lx/2, Lx/2, 0.001)

beta = 2.0
kBT = 1.0
kappa = 1.0
xl = 0.5
Np = 400
zeta = np.sqrt(2*kappa/beta)
ep = xl/(kBT*zeta)
beta_nd = beta/zeta**3
c0 = Np/(Lx*Ly*zeta*zeta)

# phi calculation
fig1, ax1 = plt.subplots(figsize=(8,4))

ax1.scatter(x, phi_load[:,sl]-np.tanh(x), color='red', s=10, label='$\\epsilon$=0')
ax1.plot(x_a, np.tanh(x_a)-np.tanh(x_a), color='red', linewidth=1)
ax1.set_xlabel('x')
ax1.set_ylabel('$\\phi(x)-tanh(x)$')

phi_a = np.tanh(x_a) + ((ep**2*0.5*c0*x_a*sech(x_a)**2) / beta_nd) + (1/16)*(ep**4/beta_nd**2)*sech(x_a)**2*c0*((-4*c0*x_a**2 + sech(x_a)**2*beta_nd + 2*beta_nd)*np.tanh(x_a) + 6*c0*x_a)
phi_load = np.loadtxt(f'./interface-surfactant-5-np-400-data/phi{n}.txt')
ax1.scatter(x, phi_load[:,sl]-np.tanh(x), color='blue', s=10, label='$\\epsilon$=0.5')
ax1.plot(x_a, phi_a-np.tanh(x_a), color='blue', linewidth=1)
ax1.legend(loc='upper right')
ax1.set_xlim([-10,10])

plt.savefig('phi-interface.jpg')
plt.close()


# c calculation
fig1, ax1 = plt.subplots(figsize=(8,4))
c_a = c0*(1 + 0.25*ep**2*sech(x_a)**4) + 0.5*(c0*ep**4)*(2*c0*np.cosh(2*x_a)**2 + 4*c0*np.cosh(2*x_a)*(1-np.sinh(2*x_a)*x_a) - 4*c0*np.sinh(2*x_a)*x_a + beta_nd + 2*c0)/(beta_nd*(1+np.cosh(2*x_a))**4)
c_a = c_a - c0

c_load = np.loadtxt(f'./interface-surfactant-5-np-400-data/c{n}.txt')
ax1.scatter(x, c_load[:,sl]-c0, color='blue', s=10, label='$\\epsilon$=0.5')
ax1.plot(x_a, c_a, color='blue', linewidth=1)
ax1.legend(loc='upper right')
ax1.set_xlim([-10,10])
ax1.set_xlabel('x')
ax1.set_ylabel('$c(x)-c_0$')

plt.savefig('c-interface.jpg')
plt.close()


# px calculation
fig1, ax1 = plt.subplots(figsize=(8,4))

px_a = ep*(-0.5 + 0.5*np.tanh(x_a)**2) + 0.25*(ep**3*c0*sech(x_a)**2 * (2*x_a*np.tanh(x_a) - 1))/beta_nd

px_load = np.loadtxt(f'./interface-surfactant-5-np-400-data/px{n}.txt')
ax1.scatter(x, px_load[:,sl], color='blue', s=10, label='$\\epsilon$=0.5')
ax1.plot(x_a, px_a, color='blue', linewidth=1)
ax1.legend(loc='upper right')
ax1.set_xlim([-10,10])
ax1.set_xlabel('x')
ax1.set_ylabel('$p_x(x)$')


plt.savefig('px-interface.jpg')
plt.close()

