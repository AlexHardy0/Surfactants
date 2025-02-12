# imports
import numpy as np
import matplotlib.pyplot as plt

# space parameters from sim
dx = 1.0 #assuming dx == dy
Nx = 64
Ny = 64
Lx = int(Nx*dx)
Ly = int(Ny*dx)

# meshgrid for plotting
x = np.arange(0.0,Lx,dx)
y = np.arange(0.0,Ly,dx)
Y,X = np.meshgrid(y,x)

# centre slice
cs = (slice(1,-1),slice(1,-1))

X = X[cs]
Y = Y[cs]

# plotting
fig,ax = plt.subplots(3,3,sharex=True, sharey=True, figsize=(11,9))

NP = 1000

times = [int(5000*1),int(5000*12),int(5000*40)]

# plot for each square
for i,n in enumerate(times):

    phi = np.loadtxt(f"ultrastable emulsion - no surfactant vrs data/phi{n}.txt")
    print(np.sum(phi[cs]*dx*dx))
    phi_plot = ax[i,0].pcolormesh(X,Y,phi[cs],cmap='plasma')

    phi = np.loadtxt(f"ultrastable emulsion 0.5 - {NP} data/phi{n}.txt")
    px =  np.loadtxt(f"ultrastable emulsion 0.5 - {NP} data/px{n}.txt")
    px = px[cs]
    py =  np.loadtxt(f"ultrastable emulsion 0.5 - {NP} data/py{n}.txt")
    py = py[cs]
    print(np.sum(phi[cs]*dx*dx))

    phi_plot = ax[i,1].pcolormesh(X,Y,phi[cs],cmap='plasma')
    ax[0,1].set_title("$\epsilon$ = 0.5")
    ax[0,0].set_title("No surfactant")
    ax[i,0].set_ylabel("Y")

    # Threshold for phi near zero
    threshold = 0.3 
    mask = np.abs(phi[cs]) < threshold  
    # Quiver plot
    ax[i,1].quiver(X[mask], Y[mask], px[mask], py[mask],angles='xy', scale_units='xy',color='white',pivot = 'middle',scale=0.05,width=0.007,linewidth=2.0)
    
    # plt.colorbar(phi_plot)

    phi = np.loadtxt(f"ultrastable emulsion 1.0 - {NP} data/phi{n}.txt")
    px =  np.loadtxt(f"ultrastable emulsion 1.0 - {NP} data/px{n}.txt")
    px = px[cs]
    py =  np.loadtxt(f"ultrastable emulsion 1.0 - {NP} data/py{n}.txt")
    py = py[cs]
    print(np.sum(phi[cs]*dx*dx))

    phi_plot = ax[i,2].pcolormesh(X,Y,phi[cs],cmap='plasma',vmax=  1.0,vmin = -1.0)
    ax[0,2].set_title("$\epsilon$ = 1.0")
    
    # Threshold for phi near zero
    threshold = 0.3  
    mask = np.abs(phi[cs]) < threshold  
    # Quiver plot
    ax[i,2].quiver(X[mask], Y[mask], px[mask], py[mask],angles='xy', scale_units='xy',color='white',pivot='middle',scale=0.1,width=0.007,linewidth=2.0)
    

ax[i,0].set_xlabel("X")
ax[i,1].set_xlabel("X")
ax[i,2].set_xlabel("X")

# Define positions for each subplot [left, bottom, width, height]
positions = [
    [0.08, 0.57, 0.27, 0.25],  # Top-left
    [0.36, 0.57, 0.27, 0.25],  # Top-center
    [0.64, 0.57, 0.27, 0.25],  # Top-right
    [0.08, 0.31, 0.27, 0.25],  # Middle-left
    [0.36, 0.31, 0.27, 0.25],  # Middle-center
    [0.64, 0.31, 0.27, 0.25],  # Middle-right
    [0.08, 0.05, 0.27, 0.25],  # Bottom-left
    [0.36, 0.05, 0.27, 0.25],  # Bottom-center
    [0.64, 0.05, 0.27, 0.25]   # Bottom-right
]

# Set each subplot's position manually
for ax, pos in zip(ax.flat, positions):
    ax.set_position(pos)

# Add a colorbar next to the subplots
cbar_ax = fig.add_axes([0.92, 0.06, 0.02, 0.75])  
fig.colorbar(phi_plot, cax=cbar_ax)

# Add labels to the side of each row
fig.text(0.02, 0.70, 'Time $t = 5000dt$', va='center', rotation='vertical', fontsize=12)
fig.text(0.02, 0.45, 'Time $t = 60000dt$', va='center', rotation='vertical', fontsize=12)
fig.text(0.02, 0.19, 'Time $t = 200000dt$', va='center', rotation='vertical', fontsize=12)

plt.show()

# plotting for just c

print("\n\n")

fig,ax = plt.subplots(figsize=(6,6))
n = int(5000*12)

c = np.loadtxt(f"ultrastable emulsion 0.5 - 1000 data/c{n}.txt")
print(np.sum(c[cs]*dx*dx))

c_plot = ax.pcolormesh(X[cs],Y[cs],c[cs],cmap='plasma')
plt.colorbar(c_plot)

ax.set_title("$\epsilon$ = 0.5")
ax.set_xlabel("X")
ax.set_ylabel("Y")

plt.show()