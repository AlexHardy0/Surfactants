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

names = ['tanh test']

phi_load = np.loadtxt('./'+names[0]+' data/phi0.txt')

#take centre of the plot
sl = int(Ny/2)
lx = (Lx-2*dx)/2
x = np.arange(-lx+dx/2,lx+dx/2,dx)

x_a = np.arange(-lx,lx,0.01)

colors = ['#1f77b4','#ff7f0e','#9467bd', '#d62728']

# #numerical plotting
fig,ax = plt.subplots()

step = 1

#phi plotting
phi_sim_1, = ax.plot(x,phi_load[1:-1,sl],'o',color=colors[0],markevery=step,label=names[0])


phi_sims = list([phi_sim_1])
#analytical plotting

beta = 2.0
kappa = 1.0

zeta = np.sqrt(2*kappa/beta)
beta_nd = beta/zeta**3

#phi plotting
ax.plot(x,np.tanh(x),'-')

ax.set(ylim=[-1.2,1.2])

# Set print options to avoid rounding
np.set_printoptions(precision=16, suppress=False)

# define a local function animate, which reads data at time step nt, and update the plot
def animate(nt):
    print(nt)
    for i,name in enumerate(names):
        # print(name)
        phi = np.loadtxt(f'./'+name+' data/phi'+str(nt)+'.txt')
        phi_sims[i].set_data(x,phi[1:-1,sl])  
 
Nt = int(1e6)
save = int(Nt / 30)

# interval = time between frames in miliseconds
# anim = animation.FuncAnimation(fig, animate, frames=range(0, Nt, save),interval = 500,blit = False,repeat = False)  
#anim.save((self.name + '.mp4'))

animate(45000)

plt.show()

