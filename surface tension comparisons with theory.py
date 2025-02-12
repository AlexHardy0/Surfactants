import numpy as np
import matplotlib.pyplot as plt

#sech function
def sech(x):
    return 1/np.cosh(x)

dx=0.25
Lx = int(512*dx)

lx = (Lx-2*dx)/2
x_a = np.arange(-lx,lx,0.01)

c0s = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
epsilons = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]


surface_tensions = np.zeros((len(epsilons),len(c0s)))

for i,ep in enumerate(epsilons):        
    for j,c0 in enumerate(c0s):

        beta_nd = 1.0

        #slice at interface
        m = int(len(x_a)/2)

        #theory
        phi_a = np.tanh(x_a) + ((ep**2*0.5*c0*x_a*sech(x_a)**2) / beta_nd) + (1/16)*(ep**4/beta_nd**2)*sech(x_a)**2*c0*((-4*c0*x_a**2 + sech(x_a)**2*beta_nd + 2*beta_nd)*np.tanh(x_a) + 6*c0*x_a)
        c_a = c0*(1 + 0.25*ep**2*sech(x_a)**4) + 0.5*(c0*ep**4)*(2*c0*np.cosh(2*x_a)**2 + 4*c0*np.cosh(2*x_a)*(1-np.sinh(2*x_a)*x_a) - 4*c0*np.sinh(2*x_a)*x_a + beta_nd + 2*c0)/(beta_nd*(1+np.cosh(2*x_a))**4)    
        px_a = ep*(-0.5 + 0.5*np.tanh(x_a)**2) + 0.25*(ep**3*c0*sech(x_a)**2 * (2*x_a*np.tanh(x_a) - 1))/beta_nd

        dphi_a_dx = np.gradient(phi_a)

        #rough calc of a
        a = (c_a/c0) * np.exp(px_a**2 + ep*px_a*dphi_a_dx)
        
        gamma=0.1
        K=0.75

        surf_ten = ((2/3)*beta_nd) + gamma*np.log(1/(1+a[m]*K))
        surface_tensions[i,j] = surf_ten


# Define a color gradient from blue to red, with 9 colors
colors = [plt.cm.get_cmap('plasma')(i / 8) for i in range(9)]

surface_tension_calcs = np.loadtxt("surface tensions.txt")

for i,ep in enumerate(epsilons):
    plt.plot(c0s,surface_tensions[i,:],color=colors[i],label=ep)
    # plt.plot(c0s,surface_tension_calcs[i,:],'-.',color=colors[i])

# plt.xscale('log')
plt.xlabel("Surfactant Concentration")
plt.ylabel("Surface Tension")
plt.legend()
plt.show()

