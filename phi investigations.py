import numpy as np
import matplotlib.pyplot as plt

# sech function
def sech(x):
    return 1/np.cosh(x)

#seeing if phi keeps changing
name = 'interface-test-clean-large-data'

Nx, Ny = 512,16
dx = 0.25
Lx, Ly = Nx*dx,Ny*dx

# meshgrid for plotting
x = np.arange(0.0,Lx,dx)
y = np.arange(0.0,Ly,dx)
Y,X = np.meshgrid(y,x)

sl = int(Ny/2)
lx = (Lx-2*dx)/2
x = np.arange(-lx+dx/2,lx+dx/2,dx)

# for Nt in np.arange(2790000,2800000,10000):
Nt = int(1e6)
phi = np.loadtxt(f'./{name}/phi{Nt}.txt')
plt.plot(x,phi[1:-1,sl]-np.tanh(x),'.',label=Nt)#

ep = 0.0
c0 = np.loadtxt(f'./{name}/c{Nt}.txt')[1,1]
beta_nd = 2.0
x_a = np.arange(-lx,lx,0.01)
phi_a = ((ep**2*c0*x_a*sech(x_a)**2) / (2*beta_nd)) #+ np.tanh(x_a) #+ # + (1/16)*(ep**4/beta_nd**2)*sech(x_a)**2*c0*((-4*c0*x_a**2 + sech(x_a)**2*beta_nd + 2*beta_nd)*np.tanh(x_a) + 6*c0*x_a)
plt.plot(x_a,phi_a)

plt.legend()
plt.show()