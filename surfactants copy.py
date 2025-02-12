#imports
import numpy as np
import matplotlib.pyplot as plt
import os

#class definition for structure
class Surfactant:
    def __init__(self,
                 dt = 0.0001, dx = 0.125, Nx = 1024, Ny = 512, Nt = int(5e5), 
                 M = 1.0, beta = 1.0, kappa = 0.5, xl = 0.1, kBT = 1.0,
                 B = 0.5, gamma_r = 1.0, gamma_t = 1.0, eta = 1.0, 
                 Np = 500,name = 'test'):
        ''' initalise every variable/array '''
        #set up folder to save data to
        self.name = name
        try:
            os.mkdir('./'+name+' data')
            self.crash = False
        except OSError:
            print("folder name in use")
            self.crash = True

        #Np
        self.Np = Np

        #time
        self.dt = dt
        self.Nt = Nt
        self.save = int(Nt / 10)

        #regular space
        self.dr = dx  #assuming dx == dy
        self.Nx = Nx
        self.Nz = Ny
        self.Lr = int(Nx*dx)
        self.Lz = int(Ny*dx)

        # meshgrid for plotting
        x = np.arange(0.0,self.Lr,self.dr)
        y = np.arange(0.0,self.Lz,self.dr)
        self.Y,self.X = np.meshgrid(y,x)

        #fourier space
        kr = np.fft.fftfreq(2*Nx,d=dx) * 2 * np.pi
        kz = np.fft.fftfreq(Ny,d=dx) * 2 * np.pi
        self.kz,self.kr = np.meshgrid(kz,kr)
        self.k2 = self.kr**2 + self.kz**2
        self.k4 = self.k2 * self.k2

        #constants
        self.M = M
        self.beta = beta
        self.kappa = kappa
        self.xl = xl
        self.B = B
        self.gamma_t = gamma_t
        self.gamma_r = gamma_r
        self.eta = eta
        self.kBT = kBT

        #scalar field array definitions
        self.phi = np.zeros((Nr,Nz))
        self.mu_phi = np.zeros((Nr,Nz))
        self.c = np.zeros((Nr,Nz))
        self.mu_c = np.zeros((Nr,Nz))
        self.dmu_c_dx = np.zeros((Nr,Nz))
        self.dmu_c_dy= np.zeros((Nr,Nz))

        #vector field array definitions
        self.u_r = np.zeros((Nr,Nz))
        self.u_z = np.zeros((Nr,Nz))
        self.u_r_k = np.zeros((2*Nr,Nz),dtype = complex)
        self.u_z_k = np.zeros((2*Nr,Nz),dtype = complex)
        self.p_r = np.zeros((Nr,Nz))
        self.p_z = np.zeros((Nr,Nz))
        self.c_p_r = np.zeros((Nr,Nz))
        self.c_p_z = np.zeros((Nr,Nz))
        self.f_r = np.zeros((Nr,Nz))
        self.f_z = np.zeros((Nr,Nz))
        self.h_r = np.zeros((Nr,Nz))
        self.h_z = np.zeros((Nr,Nz))

        #matrix array defintions
        #omega_rx amd omega_zy = 0
        self.omega_ry = np.zeros((Nx-2,Ny-2))
        self.omega_zx = np.zeros((Nx-2,Ny-2))
        self.D_rx = np.zeros((Nx-2,Ny-2))
        self.D_ry = np.zeros((Nx-2,Ny-2))
        #D_ry == D_zx
        self.D_zy = np.zeros((Nx-2,Ny-2))
        self.sigma_rx = np.zeros((Nr,Nz))
        self.sigma_ry = np.zeros((Nr,Nz))
        self.sigma_zx = np.zeros((Nr,Nz))
        self.sigma_zy = np.zeros((Nr,Nz))

        #initalise phi
        self.init_phi()
        #initialise c
        self.init_c()

    def init_phi(self):
        '''initalise phi as a flat vertical interface'''
        half = int(self.Nx/2)
        self.phi[:,:] = 1.0
        self.phi[:half,:] = -1.0
        
        #plotting of initial condition
        
        #plt.pcolormesh(self.X,self.Y,self.phi)
        #plt.show()

    def init_c(self):
        '''initalise c as uniform everywhere'''
        Lx = self.Lr - 2*self.dr
        Ly = self.Lz - 2*self.dr
        c_init = self.Np/(Lx*Ly)
        self.c[1:-1,1:-1] = c_init
        self.BC_standard(self.c)

        #plotting of initial condition
        
        #plt.pcolormesh(self.X,self.Y,self.c)
        #plt.show()

    def dev_r(self,q):
        return (q[2:,1:-1] - q[0:-2,1:-1]) / (2*self.dr)
    
    def dev_z(self,q):
        return (q[1:-1,2:] - q[1:-1,0:-2]) / (2*self.dr)
    
    def lap(self,q):
        return (q[2:,1:-1] + q[0:-2,1:-1] + q[1:-1,2:] + q[1:-1,0:-2]- 4*q[1:-1,1:-1]) / (self.dr*self.dr)
    
    def div(self,qx,qy):
        '''calculates divergence of a vector field (nabla . q) '''
        dq_dx = self.dev_r(qx)
        dq_dy = self.dev_z(qy)
        div = dq_dx + dq_dy

        return div

    def grad(self,q):
        '''calculates the gradient of a scalar field (nabla(q)) '''
        dq_dx = self.dev_r(q)
        dq_dy = self.dev_z(q)
        grad = [dq_dx,dq_dy]

        return grad

    def BC_standard(self,q):
        '''apply standard boundary conditions'''
        #standard = periodic on y walls, neumann on x walls
        q[0,:] = q[1,:]
        q[-1,:] = q[-2,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def BC_neg_standard(self,q):
        '''apply standard boundary conditions'''
        #negative standard = periodic on y walls, negative neumann on x walls
        q[0,:] = -q[1,:]
        q[-1,:] = -q[-2,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def BC_zeros(self,q):
        '''apply current boundary conditions'''
        #zeros for walls, periodic in y
        q[0,:] = 0
        q[-1,:] = 0

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def update_all(self,cs):
        #COMMONLY USED GRADIENTS
        #grad_phi
        dphi_dx,dphi_dy = self.grad(self.phi)
        #grad u_r
        du_r_dx, du_r_dy = self.grad(self.u_r)
        #grad u_z
        du_z_dx, du_z_dy = self.grad(self.u_z)
        #grad_c
        dc_dx, dc_dy = self.grad(self.c)

        #PRELIMINARY CALCULATIONS FOR UPDATED VARIABLES
        #calculate mu_phi
        self.c_p_r[cs] = self.c[cs]*self.p_r[cs]
        self.c_p_z[cs] = self.c[cs]*self.p_z[cs]
        self.BC_standard(self.c_p_r)
        self.BC_standard(self.c_p_z)
        self.mu_phi[cs] = self.beta*(self.phi[cs]**3 - self.phi[cs]) -0.5*self.kappa*self.lap(self.phi) - self.xl*self.div(self.c_p_r,self.c_p_z)
        self.BC_standard(self.mu_phi)

        #calculate 2/3 of mu_c
        self.mu_c[cs] = self.kBT*(self.p_r[cs]**2 + self.p_z[cs]**2) + self.xl*(self.p_r[cs]*dphi_dx + self.p_z[cs]*dphi_dy)
        self.BC_neg_standard(self.mu_c)

        #calculate h
        self.h_r[cs] = 2*self.kBT*self.c[cs]*self.p_r[cs] + self.xl*self.c[cs]*dphi_dx
        self.h_z[cs] = 2*self.kBT*self.c[cs]*self.p_z[cs] + self.xl*self.c[cs]*dphi_dy
        self.BC_standard(self.h_r)
        self.BC_standard(self.h_z)
        
        #calculate omega
        self.omega_ry = 0.5*(du_z_dx - du_r_dy)
        self.omega_zx = 0.5*(du_r_dy - du_z_dx)

        #calculate D
        self.D_rx = du_r_dx
        self.D_ry = 0.5*(du_z_dx + du_r_dy)
        self.D_zy = du_z_dy

        #calculate sigma
        self.sigma_rx[cs] = (self.B/2) * (self.p_r[cs]*self.h_r[cs])
        self.sigma_ry[cs] = (self.p_r[cs]*self.h_z[cs])*((self.B/4) - 0.5) + (self.p_z[cs]*self.h_r[cs])*((self.B/4)+0.5)
        self.sigma_zx[cs] = (self.p_z[cs]*self.h_r[cs])*((self.B/4) - 0.5) + (self.p_r[cs]*self.h_z[cs])*((self.B/4)+0.5)
        self.sigma_zy[cs] = (self.B/2) * (self.p_z[cs]*self.h_z[cs])
        self.BC_standard(self.sigma_rx)
        self.BC_standard(self.sigma_ry)
        self.BC_standard(self.sigma_zx)
        self.BC_standard(self.sigma_zy)

        #calculate f
        dh_r_dx, dh_r_dy = self.grad(self.h_r)
        dh_z_dx, dh_z_dy = self.grad(self.h_z)
        self.f_r[cs] = -self.phi[cs]*self.dev_r(self.mu_phi) - self.c[cs]*self.dev_r(self.mu_c) - self.kBT*dc_dx - self.p_r[cs]*dh_r_dx - self.p_z[cs]*dh_z_dx + self.dev_r(self.sigma_rx) + self.dev_z(self.sigma_ry)
        self.f_z[cs] = -self.phi[cs]*self.dev_r(self.mu_phi) - self.c[cs]*self.dev_r(self.mu_c) - self.kBT*dc_dx - self.p_r[cs]*dh_r_dy - self.p_z[cs]*dh_z_dy + self.dev_r(self.sigma_zx) + self.dev_z(self.sigma_zy)
        self.BC_zeros(self.f_r)
        self.BC_zeros(self.f_z)

        #UPDATE VARIABLES
        #update phi
        self.phi[cs] += self.dt*(-self.u_r[cs]*dphi_dx - self.u_z[cs]*dphi_dy+ self.M*self.lap(self.mu_phi)) 
        self.BC_standard(self.phi)

        #update c
        self.dmu_c_dx[cs],self.dmu_c_dy[cs] = (self.c[cs]/self.gamma_t)*self.grad(self.mu_c)
        self.BC_neg_standard(self.dmu_c_dx)
        self.BC_neg_standard(self.dmu_c_dy)
        self.c[cs] += self.dt*(-self.u_r[cs]*dc_dx - self.u_z[cs]*dc_dy + (self.kBT/self.gamma_t)*self.lap(self.c) + self.div(self.dmu_c_dx,self.dmu_c_dy))    
        self.BC_standard(self.c)   

        #update px
        self.p_r[cs] += self.dt*(-self.u_r[cs]*self.dev_r(self.p_r) - self.u_z[cs]*self.dev_z(self.p_r) + self.omega_ry*self.p_z[cs] + (self.B/2)*(self.D_rx*self.p_r[cs] + self.D_ry*self.p_z[cs])\
                                 - (1/(2*self.gamma_r*self.c[cs]))*self.h_r[cs])
        self.BC_standard(self.p_r)

        #update py
        self.p_z[cs] += self.dt*(-self.u_r[cs]*self.dev_r(self.p_z) - self.u_z[cs]*self.dev_z(self.p_z) + self.omega_zx*self.p_z[cs] + (self.B/2)*(self.D_ry*self.p_r[cs] + self.D_zy*self.p_z[cs])\
                                 - (1/(2*self.gamma_r*self.c[cs]))*self.h_z[cs])
        self.BC_standard(self.p_z)

        #create flipped f
        fx = np.zeros((2*self.Nx,self.Nz))
        fx[:self.Nx,:] = self.f_r
        fx[self.Nx:,:] = -self.f_r[::-1,:]

        fy = np.zeros((2*self.Nx,self.Nz))
        fy[:self.Nx,:] = self.f_z
        fy[self.Nx:,:] = -self.f_z[::-1,:]
        
        #fourier parameters
        fx_k = np.fft.fft2(fx,norm='ortho')*self.dr
        fy_k = np.fft.fft2(fy,norm='ortho')*self.dr

        #preliminary calculations for velocity
        k_f = (self.kr*fx_k) + (self.kz*fy_k)
    
        #to avoid a divide by 0
        ind = (self.k2 != 0)
        #index that is dodged remains zeros anyways (i think)
    
        #velocity calculations
        self.u_r_k[ind] += (self.dt)*(fx_k[ind] - self.eta*self.k2[ind]*self.u_r_k[ind] - self.kr[ind]*k_f[ind]/self.k2[ind])
        self.u_z_k[ind] += (self.dt)*(fy_k[ind] - self.eta*self.k2[ind]*self.u_z_k[ind] - self.kz[ind]*k_f[ind]/self.k2[ind])
    
        #transform u back into real space
        self.u_r = (np.fft.ifft2(self.u_r_k,norm='ortho').real/self.dr)[:self.Nx,:]
        self.u_z = (np.fft.ifft2(self.u_z_k,norm='ortho').real/self.dr)[:self.Nx,:]
        
    def run(self):
        '''running loop'''
        #indexes [1:-1,1:-1], to keep from rewriting it and taking up space
        #cs = central slice
        cs = (slice(1,-1),slice(1,-1))
        phi_con = 0
        c_con = self.Np

        #run entire sim
        for i in range(self.Nt):
            self.update_all(cs)

            #print conservations
            phi_sum = np.sum(self.phi[1:-1,1:-1]*self.dr*self.dr)
            c_sum = np.sum(self.c[1:-1,1:-1]*self.dr*self.dr)
            if np.round(phi_sum-phi_con) != 0.0:
                print("phi not conserved")
            if np.round(c_sum-c_con) != 0.0:
                print("c not conserved")
            #print(f'phi sum: {phi_sum}   c sum: {c_sum}')

            #save files
            if i % self.save == 0:
                np.savetxt(f'./'+self.name+' data/phi'+str(i)+'.txt',s.phi)
                np.savetxt(f'./'+self.name+' data/c'+str(i)+'.txt',s.c)
                np.savetxt(f'./'+self.name+' data/px'+str(i)+'.txt',s.p_r)
                #np.savetxt(f'./'+self.name+'data/py'+str(i)+'.txt',s.p_z)


#intiate class
s = Surfactant(dx = 0.25, dt = 0.001, Nt = int(15e4),name='dx_0.25')
if s.crash == False:
    s.run()

s = Surfactant(dx = 0.5, dt = 0.01, Nt = int(15e4),name='dx_0.5')
if s.crash == False:
    s.run()

s = Surfactant(dx = 1.0, dt = 0.01, Nt = int(15e4),name='dx_1.0')
if s.crash == False:
    s.run()

# #plotting
# fig,ax = plt.subplots(4,1)

# #phi plotting
# phi = ax[0].pcolormesh(s.X,s.Y,s.phi)
# plt.colorbar(phi)

# #c plotting
# c = ax[1].pcolormesh(s.X,s.Y,s.c)
# plt.colorbar(c)

# # #c plotting + arrows
# # sl = int(s.Nx/2)
# # st = 5
# # p = ax[1].quiver(s.X[sl,::st],s.Y[sl,::st],s.p_r[sl,::st],s.p_z[sl,::st],scale_units = 'x')

# #p plotting
# p_mag = np.sqrt(s.p_r**2 + s.p_z**2)
# p = ax[2].pcolormesh(s.X,s.Y,p_mag)
# #p = ax[2].quiver(s.X,s.Y,s.p_r,s.p_z)
# plt.colorbar(p)

# #velocity plotting
# x = np.arange(0.0,s.Lx,s.dx)
# y = np.arange(0.0,s.Ly,s.dx)
# Vx,Vy = np.meshgrid(x,y)

# uu = s.u_r.swapaxes(0,1)
# vv = s.u_z.swapaxes(0,1)

# vel_mag = np.sqrt(uu**2 + vv**2)
# v = ax[3].streamplot(Vx,Vy,uu,vv,color=vel_mag)
# plt.colorbar(v.lines)

# plt.show()

# #1D plotting

# #take centre of the plot
# sl = int(s.Ny/2)
# lx = (s.Lx-2*s.dx)/2
# x = np.arange(-lx,lx,s.dx)

# # #numerical plotting
# fig,ax = plt.subplots(3,1)

# #phi plotting
# ax[0].plot(x,s.phi[1:-1,sl])

# #c plotting
# ax[1].plot(x,s.c[1:-1,sl])

# #p plotting
# ax[2].plot(x,s.p_r[1:-1,sl])

# #analytical plotting

# #sech function
# def sech(x):
#     return 1/np.cosh(x)

# #epsilon calculation
# ep = s.xl/(s.kBT*np.sqrt(2*s.kappa/s.beta))

# #calc c0
# #c_expr = 1 + 0.25*ep**2*sech(x)**3
# #c0 = (s.Np/(s.dx*s.Ly))/np.sum(c_expr)
# c0 = s.c[1,1]
# print(c0)

# #phi plotting
# phi_a = np.tanh(x) + ((0.5*c0*x*sech(x)**2) / s.beta) #+ (1/16)*(ep**4/s.beta**2)*sech(x)**2*c0*((-4*c0*x**2 + sech(x)**2*s.beta + 2*s.beta)*np.tanh(x) + 6*c0*x)
# ax[0].plot(x,phi_a,'--')

# #c plotting
# c_a = c0*(1 + 0.25*ep**2*sech(x)**3) #+ 0.5*(c0*ep**4)*(2*c0*np.cosh(2*x)**2 + 4*c0*np.cosh(2*x)*(1-np.sinh(2*x)*x) - 4*c0*np.sinh(2*x)*x + s.beta + 2*c0)/(s.beta*(1+np.cosh(2*x))**4)
# ax[1].plot(x,c_a,'--')

# #px plotting
# px_a = ep*(-0.5 + 0.5*np.tanh(x)**2) #+ 0.25*(ep**3*c0*sech(x)**2 * (2*x*np.tanh(x) - 1))/s.beta
# ax[2].plot(x,px_a,'--')

# plt.show()
