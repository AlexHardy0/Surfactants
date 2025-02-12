# imports
import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
from scipy.integrate import simpson, trapezoid

#define variables
epsilon, beta, x, c0, a = sp.symbols('epsilon beta x c0 a')
phi = sp.Function('phi')(x)
c = sp.Function('c')(x)
px = sp.Function('px')(x)
# F = sp.Function('F')(phi,c,px)

# analytical solutions
# capped at second order to decrease run time
phi = sp.tanh(x) + (0.5*(epsilon**2)*c0*x*(sp.sech(x))**2 / beta)# + (1/16)*(epsilon**4/beta**2)*sp.sech(x)**2*c0*((-4*c0*x**2 + sp.sech(x)**2*beta + 2*beta)*sp.tanh(x) + 6*c0*x)

phi_pure = sp.tanh(x)

c = (1 + 0.25*epsilon**2*(sp.sech(x)**4))*c0 #+ 0.5*(c0*epsilon**4)*(2*c0*sp.cosh(2*x)**2 + 4*c0*sp.cosh(2*x)*(1-sp.sinh(2*x)*x) - 4*c0*sp.sinh(2*x)*x + beta + 2*c0)/(beta*(1+sp.cosh(2*x))**4)

px = (-0.5 + 0.5*sp.tanh(x)**2)*epsilon #+ 0.25*(epsilon**3*c0*sp.sech(x)**2 * (2*x*sp.tanh(x) - 1))/beta

# free energy expression in 1D
F = -0.5*beta*phi**2 + 0.25*beta*phi**4 + 0.25*beta*sp.diff(phi,x)**2  + c*sp.ln(c*a) + epsilon*c*px*sp.diff(phi,x) + c*px**2

F_pure = -0.5*beta*phi_pure**2 + 0.25*beta*phi_pure**4 + 0.25*beta*sp.diff(phi_pure,x)**2


# data for plotting and calculations
c0s = [0.0001,0.0002,0.0003,0.0004,0.0005,0.0006,0.0007,0.0008,0.0009,0.001,0.002,0.003,0.004,0.005,0.006,0.007,0.008,0.009,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
epsilons = [0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]

l = 2
x_range = np.arange(-l,l+0.1,0.1)

# store surface tension 
surface_tension = np.zeros((len(epsilons),len(c0s)))

# pure surface tension calculations
F_pure_num = F_pure.subs({beta: 1, phi: phi.subs({beta: 1})})
F_pure_fin = sp.simplify(F_pure_num)
F_pure_fin = F_pure_fin - F_pure_fin.subs(x,-100).evalf() 


F_lambda = sp.lambdify(x,F_pure_fin)
F_area = simpson(F_lambda(x_range),x_range)
print(F_area)

# Evaluate F_numeric for each x value
F_vals = [F_pure_fin.subs(x, val) for val in x_range]

# plotting
# plt.plot(x_range,F_vals,'--',label='PURE')

#calculate surface tension for each combination of c0 and epsilion
for i,ep in enumerate(epsilons):
    epsilon_val = ep

    for j,C0 in enumerate(c0s):
        c0_val = C0

        # Substitute numerical values for the constants
        F_numeric = F.subs({epsilon: epsilon_val, beta: 1, c0: c0_val, a: 1, phi: phi.subs({epsilon: epsilon_val, c0: c0_val, beta: 1}),c: c.subs({c0: c0_val, epsilon: epsilon_val}),
                                    px: px.subs({epsilon: epsilon_val})})

        # print(c.subs({c0: c0_val, epsilon: epsilon_val, x: 0}).evalf())
        F_final = sp.simplify(F_numeric)
        print(F_final)

        F_final = F_final - F_final.subs(x, -100).evalf() 
        F_lambda = sp.lambdify(x,F_final)

        x_range = np.arange(-l,l+0.1,0.1)
        F_area = simpson(F_lambda(x_range),x_range)
        surface_tension[i,j] = F_area

        # Evaluate F_numeric for each x value
        # F_vals = [F_final.subs(x, val) for val in x_range]

        # plotting 
        # plt.plot(x_range,F_vals,label=str(C0))

# save data for easier re-plotting 
# np.savetxt("surface tensions.txt",surface_tension)

# PLOTTING
fig,ax = plt.subplots(figsize=(6,6))

#load data
surface_tension = np.loadtxt("surface tensions.txt")

# Define a color gradient from blue to red, with 9 colors
colors = [plt.cm.get_cmap('cool')(i / 8) for i in range(9)]

# plot a surface tension vs. c0 line for each epsilon value
for i,ep in enumerate(epsilons):
    leg_label = f"$\epsilon$ = {ep}"
    ax.plot(c0s,surface_tension[i,:],linewidth=2,color=colors[i],label=leg_label)


# Increase font sizes globally
ax.set_xlabel("X Axis Label", fontsize=15)
ax.set_ylabel("Y Axis Label", fontsize=15)
ax.tick_params(axis='both', labelsize=15)
ax.legend(fontsize=15)

plt.tight_layout()
plt.xscale('log')
plt.xlabel("Surfactant Concentration")
plt.ylabel("Surface Tension")
plt.show()
