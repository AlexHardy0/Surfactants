import numpy as np
import matplotlib.pyplot as plt

## prepare things like the meshgrid, etc - will be different for each file
## manually to save time with automating

## naming convention: f = full = dx=1.0, h = half, = dx=0.5, q = quarter = dx=0.25, e = eighth = dx=0.125

#dx vaLues
f_dx = 1.0
h_dx = 0.5
q_dx = 0.25
e_dx = 0.125

#Nx and Ny values
Nx = 1024
Ny = 512

#Lx values
f_Lx = int(Nx*f_dx)
h_Lx = int(Nx*h_dx)
q_Lx = int(Nx*q_dx)
e_Lx = int(Nx*e_dx)

#bounds for the x-axis
f_bnds = (f_Lx - 2*f_dx)/2
h_bnds = (h_Lx - 2*h_dx)/2
q_bnds = (q_Lx - 2*q_dx)/2
e_bnds = (e_Lx - 2*e_dx)/2

#x-axis
f_r = np.arange(-f_bnds,f_bnds,f_dx)
h_r = np.arange(-h_bnds,h_bnds,h_dx)
q_r = np.arange(-q_bnds,q_bnds,q_dx)
e_r = np.arange(-e_bnds,e_bnds,e_dx)

#middle index for plotting
mid = int(Ny/2)

## we are just checking the convergence first
## load in the last file for each run

f_phi = np.loadtxt('dx_1.0 data/phi135000.txt')
f_c = np.loadtxt('dx_1.0 data/c135000.txt')
f_px = np.loadtxt('dx_1.0 data/px135000.txt')


h_phi = np.loadtxt('dx_0.5 data/phi135000.txt')
h_c = np.loadtxt('dx_0.5 data/c135000.txt')
h_px = np.loadtxt('dx_0.5 data/px135000.txt')


q_phi = np.loadtxt('dx_0.25 data/phi135000.txt')
q_c = np.loadtxt('dx_0.25 data/c135000.txt')
q_px = np.loadtxt('dx_0.25 data/px135000.txt')


e_phi = np.loadtxt('dx_0.125 data/phi450000.txt')
e_c = np.loadtxt('dx_0.125 data/c450000.txt')
e_px = np.loadtxt('dx_0.125 data/px450000.txt')


##plotting

fig,ax = plt.subplots(3,1)

# #plotting f
# ax[0].plot(f_r,f_phi[1:-1,mid],'r')
# ax[1].plot(f_r,f_c[1:-1,mid],'r')
# ax[2].plot(f_r,f_px[1:-1,mid],'r')

#plotting h
ax[0].plot(h_r,h_phi[1:-1,mid],'b')
ax[1].plot(h_r,h_c[1:-1,mid],'b')
ax[2].plot(h_r,h_px[1:-1,mid],'b')

# #plotting q
# ax[0].plot(q_r,q_phi[1:-1,mid],'g')
# ax[1].plot(q_r,q_c[1:-1,mid],'g')
# ax[2].plot(q_r,q_px[1:-1,mid],'g')

# #plotting e
# ax[0].plot(e_r,e_phi[1:-1,mid],'m')
# ax[1].plot(e_r,e_c[1:-1,mid],'m')
# ax[2].plot(e_r,e_px[1:-1,mid],'m')

#sech function
def sech(x):
    return 1/np.cosh(x)

#epsilon calculation
ep = 0.1/(1.0*np.sqrt(2*0.5/1.0))

#read c0 from file
c0 = h_c[1,1]
#print(c0)

#which x to use
x = h_r

#phi plotting
phi_a = np.tanh(x) + ((0.5*c0*x*sech(x)**2) / 1.0) #+ (1/16)*(ep**4/1.0**2)*sech(x)**2*c0*((-4*c0*x**2 + sech(x)**2*1.0 + 2*1.0)*np.tanh(x) + 6*c0*x)
ax[0].plot(x,phi_a,'k--')

#c plotting
c_a = c0*(1 + 0.25*ep**2*sech(x)**3) #+ 0.5*(c0*ep**4)*(2*c0*np.cosh(2*x)**2 + 4*c0*np.cosh(2*x)*(1-np.sinh(2*x)*x) - 4*c0*np.sinh(2*x)*x + 1.0 + 2*c0)/(1.0*(1+np.cosh(2*x))**4)
ax[1].plot(x,c_a,'k--')

#px plotting
px_a = ep*(-0.5 + 0.5*np.tanh(x)**2) #+ 0.25*(ep**3*c0*sech(x)**2 * (2*x*np.tanh(x) - 1))/1.0
ax[2].plot(x,px_a,'k--')

plt.show()