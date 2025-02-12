import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

Nx, Ny = 512,256
dx = 0.25
Lx, Ly = Nx*dx,Ny*dx

# meshgrid for plotting
x = np.arange(0.0,Lx,dx)
y = np.arange(0.0,Ly,dx)
Y,X = np.meshgrid(y,x)

names = ['epsilon=0.5']
# cs = (slice(1,-1),slice(1,-1))

c_load = np.loadtxt('./'+names[0]+' data/phi0.txt')
px_load = np.loadtxt('./'+names[0]+' data/px0.txt')
py_load = np.loadtxt('./'+names[0]+' data/py0.txt')

#take centre of the plot
x = np.arange(0.0,Lx,dx)
y = np.arange(0.0,Ly,dx)
Y,X = np.meshgrid(y,x)

# #numerical plotting
fig1,ax1 = plt.subplots(figsize=(6,6))

#c plotting
c_sim_1 = ax1.pcolormesh(X,Y,c_load,cmap='plasma',label=names[0])
plt.colorbar(c_sim_1)

#p plotting
px_sim_1 = ax1.quiver(X,Y,px_load,py_load,scale = 7.0)

# Increase font sizes globally
ax1.set_xlabel("X", fontsize=15)
ax1.set_ylabel("$Y$", fontsize=15)
ax1.tick_params(axis='both', labelsize=15)
ax1.set_title("$\epsilon = 0.5$",fontsize=15)
# ax1.legend(fontsize=13,frameon=False)

ax1.set(ylim=(29,31),xlim=(62.25,65.5))

# Set print options to avoid rounding
# np.set_printoptions(precision=16, suppress=False)

# define a local function animate, which reads data at time step nt, and update the plot
def animate(nt):
    print(nt)

    c = np.loadtxt(f'./'+names[0]+' data/phi'+str(nt)+'.txt')
    c_sim_1.set_array(c)

    px = np.loadtxt(f'./'+names[0]+' data/px'+str(nt)+'.txt')
    py = np.loadtxt(f'./'+names[0]+' data/py'+str(nt)+'.txt')
    px_sim_1.set_UVC(px, py)

animate(550000)
fig1.tight_layout()
plt.show()

