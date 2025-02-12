import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

Nx, Ny = 512,256
dx = 0.5
Lx, Ly = Nx*dx,Ny*dx

# meshgrid for plotting
x = np.arange(0.0,Lx,dx)
y = np.arange(0.0,Ly,dx)
Y,X = np.meshgrid(y,x)

#sech function
def sech(x):
    return 1/np.cosh(x)

name = 'epsilon = 0.1'

# phi_load = np.loadtxt('./'+name+' data/phi0.txt')
# c_load = np.loadtxt('./'+name+' data/c0.txt')
# px_load = np.loadtxt('./'+name+' data/px0.txt')

#take centre of the plots
sl = int(Ny/2)
lx = (Lx-2*dx)/2
x = np.arange(-lx+dx/2,lx+dx/2,dx)
x_a = np.arange(-lx+dx,lx+dx,0.01)

colors = ['#1f77b4','#ff7f0e','#9467bd', '#d62728']

# #numerical plotting
fig,ax = plt.subplots(3,1)

step = 1

# #phi plotting
# phi_sim_1, = ax[0].plot(x,phi_load[1:-1,sl],'o',markevery=step)

# #c plotting
# c_sim_1, = ax[1].plot(x,c_load[1:-1,sl],'o',markevery=step)

# #p plotting
# px_sim_1, = ax[2].plot(x,px_load[1:-1,sl],'o',markevery=step)


#analytical plotting

xl = 0.9
c0 = 0.015438982967145354
beta = 1.0
kBT = 1.0
kappa = 0.5

#epsilon calculation
zeta = np.sqrt(2*kappa/beta)
ep = xl/(kBT*zeta)
beta = zeta**3 * beta / kBT

#phi plotting
phi_a = np.tanh(x_a) + ((ep**2*0.5*c0*x_a*sech(x_a)**2) / beta)# + (1/16)*(ep**4/beta**2)*sech(x_a)**2*c0*((-4*c0*x_a**2 + sech(x_a)**2*beta + 2*beta)*np.tanh(x_a) + 6*c0*x_a)
ax[0].plot(x_a,phi_a)
ax[0].plot(x,np.tanh(x),'-.')

#c plotting
c_a = c0*(1 + 0.25*ep**2*sech(x_a)**4) #+ 0.5*(c0*ep**4)*(2*c0*np.cosh(2*x_a)**2 + 4*c0*np.cosh(2*x_a)*(1-np.sinh(2*x_a)*x_a) - 4*c0*np.sinh(2*x_a)*x_a + beta + 2*c0)/(beta*(1+np.cosh(2*x_a))**4)
ax[1].plot(x_a,c_a)

#px plotting
px_a = ep*(-0.5 + 0.5*np.tanh(x_a)**2) #+ 0.25*(ep**3*c0*sech(x_a)**2 * (2*x_a*np.tanh(x_a) - 1))/beta
ax[2].plot(x_a,px_a)


ax[0].set(ylim=[-1.2,1.2])
ax[1].set(ylim=[0.01542,0.0155])
ax[2].set(ylim=[-0.07,0.01])

plt.show()

# Set print options to avoid rounding
np.set_printoptions(precision=16, suppress=False)

# define a local function animate, which reads data at time step nt, and update the plot
def animate(nt):
    print(nt)
    # print(name)
    phi = np.loadtxt(f'./'+name+' data/phi'+str(nt)+'.txt')
    phi_sim_1.set_data(x,phi[1:-1,sl])  

    c = np.loadtxt(f'./'+name+' data/c'+str(nt)+'.txt')
    c_sim_1.set_data(x,c[1:-1,sl])  

    px = np.loadtxt(f'./'+name+' data/px'+str(nt)+'.txt')
    px_sim_1.set_data(x,px[1:-1,sl])

    # print(c[0:10,sl])  
    # print(c[1,1])
    
Nt = int(1e6)
save = int(Nt / 30)

# interval = time between frames in miliseconds
# anim = animation.FuncAnimation(fig, animate, frames=range(0, Nt, save),interval = 500,blit = False,repeat = False)  
# anim.save((self.name + '.mp4'))

animate(999990)

plt.show()

#PLOTTING C CHEMICAL POTENTIAL

phi_load = np.loadtxt('./'+name+' data/phi999990.txt')
c_load = np.loadtxt('./'+name+' data/c999990.txt')
px_load = np.loadtxt('./'+name+' data/px999990.txt')


def dev_x(q,dx):
    return (q[2:,1:-1] - q[0:-2,1:-1]) / (2*dx)

#calculate mu_c
xl = 0.1

dphi_dx = dev_x(phi_load,dx)
mu_c = kBT*np.log(c_load[1:-1,sl]) + kBT*px_load[1:-1,sl]**2 + xl*px_load[1:-1,sl]*dphi_dx[:,sl]

# plt.plot(mu_c)
# # plt.plot(kBT*np.log(c_load[1:-1,sl]) + kBT*px_load[1:-1,sl]**2,label='log term')
# # plt.plot(kBT*px_load[1:-1,sl]**2,label='px term')
# # plt.plot(xl*px_load[1:-1,sl]*dphi_dx[:,sl], label='dphi dx term')
# plt.legend()
# plt.show()

#plotting in 2d around the interface
phi_load = np.loadtxt('./'+name+' data/phi999990.txt')
c_load = np.loadtxt('./'+name+' data/c999990.txt')
px_load = np.loadtxt('./'+name+' data/px999990.txt')

fig,ax = plt.subplots(3,1)

interface = (slice(200,312),slice(1,-1))

phi_plot = ax[0].pcolormesh(X[interface],Y[interface],phi_load[interface])
c_plot = ax[1].pcolormesh(X[interface],Y[interface],c_load[interface])
px_plot = ax[2].pcolormesh(X[interface],Y[interface],px_load[interface])

plt.colorbar(phi_plot)
plt.colorbar(c_plot)
plt.colorbar(px_plot)

plt.show()

#PLOTTING THE INTERFACE DISTRIBUTION

phi_load = np.loadtxt('./'+name+' data/phi999990.txt')
c_load = np.loadtxt('./'+name+' data/c999990.txt')
px_load = np.loadtxt('./'+name+' data/px999990.txt')

interface = int((Nx-2*dx)/2)
# print(x[interface])

# fig,ax = plt.subplots(3,1)

# ax[0].plot(phi_load[interface,1:-1],'b')
# ax[1].plot(c_load[interface,1:-1],'b')
# ax[2].plot(px_load[interface,1:-1],'b')

# interface = int((Nx-2*dx)/2 - 1)
# print(x[interface])

# ax[0].plot(phi_load[interface,1:-1],'r')
# ax[1].plot(c_load[interface,1:-1],'r')
# ax[2].plot(px_load[interface,1:-1],'r')

# plt.show()