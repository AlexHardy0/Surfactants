# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# spatial parameters from sims
Nx, Ny = 512,256
dx = 0.25
Lx, Ly = Nx*dx,Ny*dx

# meshgrid for plotting
x = np.arange(0.0,Lx,dx)
y = np.arange(0.0,Ly,dx)
Y,X = np.meshgrid(y,x)

# sech function
def sech(x):
    return 1/np.cosh(x)

# folder names
names = ['epsilon=0.01','epsilon=0.1','epsilon=0.3','epsilon=0.5','epsilon=0.7','epsilon=0.9']

#load 0 first
phi_load = np.loadtxt('./'+names[0]+' data/phi0.txt')
c_load = np.loadtxt('./'+names[0]+' data/c0.txt')
px_load = np.loadtxt('./'+names[0]+' data/px0.txt')

# take centre of the plot
sl = int(Ny/2)
lx = (Lx-2*dx)/2
x = np.arange(-lx+dx/2,lx+dx/2,dx)

# higher def. x range for analytics
x_a = np.arange(-lx,lx,0.01)
print(x_a[6400])

colors = [plt.cm.get_cmap('cool')(i / 5) for i in range(6)]

# #numerical plotting
fig1,ax1 = plt.subplots(figsize=(8,4))
fig2,ax2 = plt.subplots(2,1,figsize=(8,8))

step = 1

#phi plotting
phi_sim_1, = ax1.plot(x,phi_load[1:-1,sl],'o',color=colors[0],markevery=step,label=names[0])
phi_sim_2, = ax1.plot(x,phi_load[1:-1,sl],'^',color=colors[1],markevery=step,label=names[1])
phi_sim_3, = ax1.plot(x,phi_load[1:-1,sl],'s',color=colors[2],markevery=step,label=names[2])
phi_sim_4, = ax1.plot(x,phi_load[1:-1,sl],'p',color=colors[3],markevery=step,label=names[3])
phi_sim_5, = ax1.plot(x,phi_load[1:-1,sl],'D',color=colors[4],markevery=step,label=names[4])
phi_sim_6, = ax1.plot(x,phi_load[1:-1,sl],'v',color=colors[5],markevery=step,label=names[5])

# ax[0].legend()

#c plotting
c_sim_1, = ax2[0].plot(x,c_load[1:-1,sl],'o',color=colors[0],markevery=step,label=names[0])
c_sim_2, = ax2[0].plot(x,c_load[1:-1,sl],'^',color=colors[1],markevery=step,label=names[1])
c_sim_3, = ax2[0].plot(x,c_load[1:-1,sl],'s',color=colors[2],markevery=step,label=names[2])
c_sim_4, = ax2[0].plot(x,c_load[1:-1,sl],'p',color=colors[3],markevery=step,label=names[3])
c_sim_5, = ax2[0].plot(x,c_load[1:-1,sl],'D',color=colors[4],markevery=step,label=names[4])
c_sim_6, = ax2[0].plot(x,c_load[1:-1,sl],'v',color=colors[5],markevery=step,label=names[5])

#p plotting
px_sim_1, = ax2[1].plot(x,px_load[1:-1,sl],'o',color=colors[0],markevery=step)
px_sim_2, = ax2[1].plot(x,px_load[1:-1,sl],'^',color=colors[1],markevery=step)
px_sim_3, = ax2[1].plot(x,px_load[1:-1,sl],'s',color=colors[2],markevery=step)
px_sim_4, = ax2[1].plot(x,px_load[1:-1,sl],'p',color=colors[3],markevery=step)
px_sim_5, = ax2[1].plot(x,px_load[1:-1,sl],'D',color=colors[4],markevery=step)
px_sim_6, = ax2[1].plot(x,px_load[1:-1,sl],'v',color=colors[5],markevery=step)


phi_sims = list([phi_sim_1,phi_sim_2,phi_sim_3,phi_sim_4,phi_sim_5,phi_sim_6])
c_sims = list([c_sim_1,c_sim_2,c_sim_3,c_sim_4,c_sim_5,c_sim_6])
px_sims = list([px_sim_1,px_sim_2,px_sim_3,px_sim_4,px_sim_5,px_sim_6])


# analytical plotting
xls = [0.01,0.1,0.3,0.5,0.7,0.9]
c0s = [0.049405576228111564,0.049404309940930195,0.04939399290079461,0.049372894244983936,0.04934001220471028,0.04929362957209666]

beta = 2.0
kBT = 1.0
kappa = 1.0


for i,(xl,c0) in enumerate(zip(xls,c0s)):
    # non dim constants calculation
    zeta = np.sqrt(2*kappa/beta)
    ep = xl/(kBT*zeta)
    beta_nd = beta/zeta**3

    # phi calculation
    phi_a = np.tanh(x_a) + ((ep**2*0.5*c0*x_a*sech(x_a)**2) / beta_nd) + (1/16)*(ep**4/beta_nd**2)*sech(x_a)**2*c0*((-4*c0*x_a**2 + sech(x_a)**2*beta_nd + 2*beta_nd)*np.tanh(x_a) + 6*c0*x_a)
    phi_a = phi_a - np.tanh(x_a)
    
    ax1.plot(x_a,phi_a,color = colors[i])
    # ax1.plot(x,np.tanh(x),'-.')

    # c calculation
    c_a = c0*(1 + 0.25*ep**2*sech(x_a)**4) + 0.5*(c0*ep**4)*(2*c0*np.cosh(2*x_a)**2 + 4*c0*np.cosh(2*x_a)*(1-np.sinh(2*x_a)*x_a) - 4*c0*np.sinh(2*x_a)*x_a + beta_nd + 2*c0)/(beta_nd*(1+np.cosh(2*x_a))**4)
    c_a = c_a - c0
    ax2[0].plot(x_a,c_a,color=colors[i])

    # px calculation
    px_a = ep*(-0.5 + 0.5*np.tanh(x_a)**2) + 0.25*(ep**3*c0*sech(x_a)**2 * (2*x_a*np.tanh(x_a) - 1))/beta_nd
    ax2[1].plot(x_a,px_a,color=colors[i])


# axis limits

# ax1.set(ylim=[-1.2,1.2],xlim=[-2,2])
# ax2[0].set(ylim=[0.049,0.064],xlim=[-2,2])
# ax2[1].set(ylim=[-0.5,0.05],xlim=[-2,2])

ax1.set(ylim=[-0.01,0.01],xlim=[-2,2])
ax2[0].set(ylim=[-0.0001,0.016],xlim=[-2,2])
ax2[1].set(ylim=[-0.5,0.05],xlim=[-2,2])

# Increase font sizes globally
ax1.set_xlabel("x", fontsize=15)
ax1.set_ylabel("$\phi$", fontsize=15)
ax1.tick_params(axis='both', labelsize=15)
ax1.legend(fontsize=13,frameon=False)

# ax2[0].set_xlabel("x", fontsize=15)
ax2[0].set_ylabel("c", fontsize=15)
ax2[0].tick_params(axis='both', labelsize=15)
ax2[0].legend(fontsize=13,frameon=False)

ax2[1].set_xlabel("x", fontsize=15)
ax2[1].set_ylabel("$p_x$", fontsize=15)
ax2[1].tick_params(axis='both', labelsize=15)
# ax[2].legend(fontsize=26)

# Set print options to avoid rounding
# np.set_printoptions(precision=16, suppress=False)

# define a local function animate, which reads data at time step nt, and update the plot
def animate(nt):
    # print(nt)
    for i,name in enumerate(names):
        # print(name)
        phi = np.loadtxt(f'./'+name+' data/phi'+str(nt)+'.txt')
        phi_sims[i].set_data(x,phi[1:-1,sl]- np.tanh(x))# 

        c = np.loadtxt(f'./'+name+' data/c'+str(nt)+'.txt')
        c_sims[i].set_data(x,c[1:-1,sl]- c0s[i] )#

        px = np.loadtxt(f'./'+name+' data/px'+str(nt)+'.txt')
        px_sims[i].set_data(x,px[1:-1,sl])

        # uncomment to get values of c0 from the sim
        # print(c[1,1])

# create animation    
# Nt = int(1e6)
# save = int(Nt / 30)

# interval = time between frames in miliseconds
# anim = animation.FuncAnimation(fig, animate, frames=range(0, Nt, save),interval = 500,blit = False,repeat = False)  
# anim.save((self.name + '.mp4'))

# create one plot
animate(1050000)

fig1.tight_layout()
fig2.tight_layout()
plt.show()

