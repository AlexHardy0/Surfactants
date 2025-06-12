# imports
import numpy as np
import matplotlib.pyplot as plt

# space parameters from sim
dx = 0.5 #assuming dx == dy
Nx = 128
Ny = 128
Lx = int(Nx*dx)
Ly = int(Ny*dx)

# meshgrid for plotting
x = np.arange(0.0,Lx,dx)
y = np.arange(0.0,Ly,dx)
Y,X = np.meshgrid(y,x)

# centre slice
cs = (slice(1,-1),slice(1,-1))

# plotting
fig,ax = plt.subplots(3,2,sharex=True, sharey=True, figsize=(11,9))

NP = 1000

times = [int(1e3*1),int(1e3*1000),int(1e3*2000)]

# plot for each square
for i,n in enumerate(times):

    phi = np.load(f"./emulsion no surf/step_{n}.npz")['phi']
    # print(np.sum(phi[cs]*dx*dx))
    phi_plot = ax[i,0].pcolormesh(X,Y,phi[cs],cmap='plasma')

    # phi = np.loadtxt(f"ultrastable emulsion 0.5 - {NP} data/phi{n}.txt")
    # px =  np.loadtxt(f"ultrastable emulsion 0.5 - {NP} data/px{n}.txt")
    # px = px[cs]
    # py =  np.loadtxt(f"ultrastable emulsion 0.5 - {NP} data/py{n}.txt")
    # py = py[cs]
    # print(np.sum(phi[cs]*dx*dx))

    # phi_plot = ax[i,1].pcolormesh(X,Y,phi[cs],cmap='plasma')
    # ax[0,1].set_title("$\epsilon$ = 0.5")
    ax[0,0].set_title("No surfactant")
    # ax[i,0].set_ylabel("Y")

    # # Threshold for phi near zero
    # threshold = 0.3 
    # mask = np.abs(phi[cs]) < threshold  
    # # Quiver plot
    # ax[i,1].quiver(X[mask], Y[mask], px[mask], py[mask],angles='xy', scale_units='xy',color='white',pivot = 'middle',scale=0.05,width=0.007,linewidth=2.0)
    
    # # plt.colorbar(phi_plot)

    data = np.load(f"./emulsion ys surf/step_{n}.npz")
    phi = data['phi']
    px = data['px']
    px = px[cs]
    py = data['py']
    py = py[cs]
    # print(np.sum(phi[cs]*dx*dx))

    phi_plot = ax[i,1].pcolormesh(X,Y,phi[cs],cmap='plasma',vmax=  1.0,vmin = -1.0)
    ax[0,1].set_title("surfactant")
    
    # Threshold for phi near zero
    threshold = 0.07  
    mask = np.abs(phi[cs]) <= threshold  
    # Quiver plot
    ax[i,1].quiver(X[mask], Y[mask], px[mask], py[mask],angles='xy', scale_units='xy',color='white',pivot='middle',scale=0.13,width=0.007,linewidth=1.0)
    

# ax[i,0].set_xlabel("X")
# ax[i,1].set_xlabel("X")

# Define positions for each subplot [left, bottom, width, height]
positions = [
    [0.08, 0.59, 0.25, 0.25],  # Top-left
    [0.35, 0.59, 0.25, 0.25],  # Top-center
    # [0.64, 0.57, 0.27, 0.25],  # Top-right
    [0.08, 0.32, 0.25, 0.25],  # Middle-left
    [0.35, 0.32, 0.25, 0.25],  # Middle-center
    # [0.64, 0.31, 0.27, 0.25],  # Middle-right
    [0.08, 0.05, 0.25, 0.25],  # Bottom-left
    [0.35, 0.05, 0.25, 0.25],  # Bottom-center
    # [0.64, 0.05, 0.27, 0.25]   # Bottom-right
]

# Set each subplot's position manually
for ax, pos in zip(ax.flat, positions):
    ax.set_position(pos)

# Add a colorbar next to the subplots
cbar_ax = fig.add_axes([0.62, 0.07, 0.02, 0.75])  
fig.colorbar(phi_plot, cax=cbar_ax)

# Add labels to the side of each row
fig.text(0.02, 0.70, 'Time $t = 0.1$', va='center', rotation='vertical', fontsize=12)
fig.text(0.02, 0.45, 'Time $t = 100$', va='center', rotation='vertical', fontsize=12)
fig.text(0.02, 0.19, 'Time $t = 200$', va='center', rotation='vertical', fontsize=12)

plt.show()

# plotting for just c

print("\n\n")

fig,ax = plt.subplots(figsize=(6,6))
n = int(10000*100)

c = data['c']
print(np.sum(c[cs]*dx*dx))

c_plot = ax.pcolormesh(X,Y,c[cs],cmap='plasma')
plt.colorbar(c_plot)

# ax.set_title("$\epsilon$ = 0.5")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.show()