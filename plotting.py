# imports
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# spatial parameters from sims
Nx, Ny = 256,8
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
names = ['interface-no-surfactant','interface-surfactant-3-np-200','interface-surfactant-5-np-400']
# names = ['speed test']

#load 0 first
# phi_load = np.loadtxt('./'+names[0]+' data/phi0.txt')
# c_load = np.loadtxt('./'+names[0]+' data/c0.txt')
# px_load = np.loadtxt('./'+names[0]+' data/px0.txt')

# Load the saved .npz file
# data = np.load(f'./{names[0]}-data/step_{0}.npz')

# Access the arrays inside the .npz file
# phi_load = data['phi']
# c_load = data['c']
# px_load = data['px']
# py_load = data['py']
phi_load = np.loadtxt(f'./{names[0]}-data/phi{0}.txt')
c_load = np.loadtxt(f'./{names[0]}-data/c{0}.txt')
px_load = np.loadtxt(f'./{names[0]}-data/px{0}.txt')
py_load = np.loadtxt(f'./{names[0]}-data/py{0}.txt')

# take centre of the plot
sl = int(Ny/2)
lx = (Lx-2*dx)/2
x = np.arange(-lx+dx/2,lx+dx/2,dx)

# higher def. x range for analytics
x_a = np.arange(-lx,lx,0.01)
# print(x_a[6400])

colors = [plt.cm.get_cmap('cool')(i / 2) for i in range(3)]

# #numerical plotting
fig1,ax1 = plt.subplots(figsize=(8,6))
fig2,ax2 = plt.subplots(figsize=(8,4))
fig3,ax3 = plt.subplots(figsize=(8,4))

# analytical plotting
xls = [0.0,0.3,0.5]
c0s = [0.0,1.561743737366377,3.123428690684442]

step = 1

#phi plotting
phi_sim_1, = ax1.plot(x,phi_load[1:-1,sl],'o',color=colors[0],markevery=step,label=f"$\epsilon$={xls[0]} c0={c0s[0]:.1f}")
phi_sim_2, = ax1.plot(x,phi_load[1:-1,sl],'^',color=colors[1],markevery=step,label=f"$\epsilon$={xls[1]} c0={c0s[1]:.1f}")
phi_sim_3, = ax1.plot(x,phi_load[1:-1,sl],'s',color=colors[2],markevery=step,label=f"$\epsilon$={xls[2]} c0={c0s[2]:.1f}")
# phi_sim_4, = ax1.plot(x,phi_load[1:-1,sl],'p',color=colors[3],markevery=step,label=f"$\epsilon$={xls[3]} c0={c0s[3]:.1f}")
# phi_sim_5, = ax1.plot(x,phi_load[1:-1,sl],'D',color=colors[4],markevery=step,label=f"$\epsilon$={xls[4]} c0={c0s[4]:.1f}")
# phi_sim_6, = ax1.plot(x,phi_load[1:-1,sl],'v',color=colors[5],markevery=step,label=f"$\epsilon$={xls[5]} c0={c0s[5]:.1f}")

# ax[0].legend()

#c plotting
c_sim_1, = ax2.plot(x,c_load[1:-1,sl],'o',color=colors[0],markevery=step,label=f"$\epsilon$={xls[0]} c0={c0s[0]:.1f}")
c_sim_2, = ax2.plot(x,c_load[1:-1,sl],'^',color=colors[1],markevery=step,label=f"$\epsilon$={xls[1]} c0={c0s[1]:.1f}")
c_sim_3, = ax2.plot(x,c_load[1:-1,sl],'s',color=colors[2],markevery=step,label=f"$\epsilon$={xls[2]} c0={c0s[2]:.1f}")
# c_sim_4, = ax2.plot(x,c_load[1:-1,sl],'p',color=colors[3],markevery=step,label=f"$\epsilon$={xls[3]} c0={c0s[3]:.1f}")
# c_sim_5, = ax2.plot(x,c_load[1:-1,sl],'D',color=colors[4],markevery=step,label=f"$\epsilon$={xls[4]} c0={c0s[4]:.1f}")
# c_sim_6, = ax2.plot(x,c_load[1:-1,sl],'v',color=colors[5],markevery=step,label=f"$\epsilon$={xls[5]} c0={c0s[5]:.1f}")

#p plotting
px_sim_1, = ax3.plot(x,px_load[1:-1,sl],'o',color=colors[0],markevery=step,label=f"$\epsilon$={xls[0]} c0={c0s[0]:.1f}")
px_sim_2, = ax3.plot(x,px_load[1:-1,sl],'^',color=colors[1],markevery=step,label=f"$\epsilon$={xls[1]} c0={c0s[1]:.1f}")
px_sim_3, = ax3.plot(x,px_load[1:-1,sl],'s',color=colors[2],markevery=step,label=f"$\epsilon$={xls[2]} c0={c0s[2]:.1f}")
# px_sim_4, = ax3.plot(x,px_load[1:-1,sl],'p',color=colors[3],markevery=step,label=f"$\epsilon$={xls[3]} c0={c0s[3]:.1f}")
# px_sim_5, = ax3.plot(x,px_load[1:-1,sl],'D',color=colors[4],markevery=step,label=f"$\epsilon$={xls[4]} c0={c0s[4]:.1f}")
# px_sim_6, = ax3.plot(x,px_load[1:-1,sl],'v',color=colors[5],markevery=step,label=f"$\epsilon$={xls[5]} c0={c0s[5]:.1f}")


phi_sims = list([phi_sim_1,phi_sim_2,phi_sim_3])
c_sims = list([c_sim_1,c_sim_2,c_sim_3])
px_sims = list([px_sim_1,px_sim_2,px_sim_3])
# phi_sims = list([phi_sim_1])
# c_sims = list([c_sim_1])
# px_sims = list([px_sim_1])

# c0s = [0.0,3.123428690684442,3.1196578381532647]
# c0s = [0.0,6.246576168927183,6.229189574951451]
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
    ax2.plot(x_a,c_a,color=colors[i])

    # px calculation
    if i in [0,1,2]:
        px_a = ep*(-0.5 + 0.5*np.tanh(x_a)**2) + 0.25*(ep**3*c0*sech(x_a)**2 * (2*x_a*np.tanh(x_a) - 1))/beta_nd
        ax3.plot(x_a,px_a,color=colors[i])


# axis limits

# ax1.set(ylim=[-1.2,1.2],xlim=[-2,2])
ax1.set(ylim=[-0.15,0.15],xlim=[-4,4])
# ax2[0].set(ylim=[0.049,0.064],xlim=[-2,2])
# ax2[0].set(ylim=[0.04,0.064],xlim=[-2,2])
ax2.set(ylim=[-0.05,0.32],xlim=[-4,4])
ax3.set(ylim=[-0.32,0.05],xlim=[-4,4])

# ax1.set(ylim=[-0.01,0.01],xlim=[-2,2])
# ax2[0].set(ylim=[-0.0001,0.016],xlim=[-2,2])
# ax2[1].set(ylim=[-0.5,0.05],xlim=[-2,2])

# Increase font sizes globally
ax1.set_xlabel("x", fontsize=15)
ax1.set_ylabel("$\phi$ - $\phi_0$", fontsize=15)
ax1.tick_params(axis='both', labelsize=15)
ax1.legend(fontsize=13,frameon=False,loc='upper left')

ax2.set_xlabel("x", fontsize=15)
ax2.set_ylabel("c - $c_0$", fontsize=15)
ax2.tick_params(axis='both', labelsize=15)
ax2.legend(fontsize=13,frameon=False)

ax3.set_xlabel("x", fontsize=15)
ax3.set_ylabel("$p_x$", fontsize=15)
ax3.tick_params(axis='both', labelsize=15)
ax3.legend(fontsize=13,frameon=False)
# ax[2].legend(fontsize=26)

# Set print options to avoid rounding
# np.set_printoptions(precision=16, suppress=False)

# define a local function animate, which reads data at time step nt, and update the plot
nts = [int(3e6),int(5e6),int(9e6),int(5e6),int(5e6),int(9e6)]

def animate(nt):
    # print(nt)
    px_i = 0
    for i,name in enumerate(names):
        nt = nts[i]
        # print(name)
        # Load the saved .npz file
        # data = np.load(f'./{name}-data/step_{i}.npz')

        # # Access the arrays inside the .npz file
        # phi = data['phi']
        # c = data['c']
        # px = data['px']

        phi = np.loadtxt(f'./{name}-data/phi{nt}.txt')
        c = np.loadtxt(f'./{name}-data/c{nt}.txt')
        px = np.loadtxt(f'./{name}-data/px{nt}.txt')
        py = np.loadtxt(f'./{name}-data/py{nt}.txt')

        
        # phi = np.loadtxt(f'./'+name+' data/phi'+str(nt)+'.txt')
        phi_sims[i].set_data(x ,phi[1:-1,sl]- np.tanh(x))#

        # c = np.loadtxt(f'./'+name+' data/c'+str(nt)+'.txt')
        c_sims[i].set_data(x,c[1:-1,sl]- c0s[i])#

        # px = np.loadtxt(f'./'+name+' data/px'+str(nt)+'.txt')
        # if i in [0,1,3]:
            # px_sims[px_i].set_data(x,px[1:-1,sl])
            # px_i+=1
        px_sims[i].set_data(x,px[1:-1,sl])

        print(c[1,1])

        # uncomment to get values of c0 from the sim

        # print(np.sum(c[1:-1,1:-1]*dx*dx))

# create animation    
Nt = int(1e6)
save = int(1e4)

# interval = time between frames in miliseconds
# anim = animation.FuncAnimation(fig2, animate, frames=range(0, Nt, save),interval = 500,blit = False,repeat = False)  
# anim.save((self.name + '.mp4'))

# create one plot
animate(int(3e6))

fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
plt.show()

