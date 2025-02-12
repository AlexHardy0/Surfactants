#imports
import numpy as np
import matplotlib.pyplot as plt
import os

#class definition for structure
class Surfactant:
    def __init__(self,
                 dt = 0.01, dx = 0.5, Nx = 512, Ny = 256, Nt = int(1e6), 
                 M = 1.0, beta = 1.0, kappa = 0.5, xl = 0.1, kBT = 1.0,
                 B = 0.5, gamma_r = 1.0, gamma_t = 1.0, eta = 1.0, 
                 Np = 500,name = 'new dx_0.5',load=False, load_no = 0, init_type = 1):
        ''' initalise every variable/array '''
        #set up folder to save data to
        self.name = name
        self.crash = False
        self.load_no = load_no
        try:
            os.mkdir('./'+name+' data')
        except OSError:
            print("folder name in use")
            if load == False:
                self.crash = True

        #Np
        self.Np = Np

        #time
        self.dt = dt
        self.Nt = Nt
        self.save = int(Nt / 30)

        #regular space
        self.dx = dx  #assuming dx == dy
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = int(Nx*dx)
        self.Ly = int(Ny*dx)

        # meshgrid for plotting
        x = np.arange(0.0,self.Lx,self.dx)
        y = np.arange(0.0,self.Ly,self.dx)
        self.Y,self.X = np.meshgrid(y,x)

        #fourier space
        kx = np.fft.fftfreq(2*Nx,d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny,d=dx) * 2 * np.pi
        self.KY,self.KX = np.meshgrid(ky,kx)
        self.k2 = self.KX**2 + self.KY**2
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
        self.phi = np.zeros((Nx,Ny))
        self.mu_phi = np.zeros((Nx,Ny))
        self.c = np.zeros((Nx,Ny))
        self.mu_c = np.zeros((Nx,Ny))
        self.dmu_c_dx = np.zeros((Nx,Ny))
        self.dmu_c_dy= np.zeros((Nx,Ny))

        #vector field array definitions
        self.u_x = np.zeros((Nx,Ny))
        self.u_y = np.zeros((Nx,Ny))
        self.u_x_k = np.zeros((2*Nx,Ny),dtype = complex)
        self.u_y_k = np.zeros((2*Nx,Ny),dtype = complex)
        self.p_x = np.zeros((Nx,Ny))
        self.p_y = np.zeros((Nx,Ny))
        self.c_p_x = np.zeros((Nx,Ny))
        self.c_p_y = np.zeros((Nx,Ny))
        self.f_x = np.zeros((Nx,Ny))
        self.f_y = np.zeros((Nx,Ny))
        self.h_x = np.zeros((Nx,Ny))
        self.h_y = np.zeros((Nx,Ny))

        #matrix array defintions
        #omega_xx amd omega_yy = 0
        self.omega_xy = np.zeros((Nx-2,Ny-2))
        self.omega_yx = np.zeros((Nx-2,Ny-2))
        self.D_xx = np.zeros((Nx-2,Ny-2))
        self.D_xy = np.zeros((Nx-2,Ny-2))
        #D_xy == D_yx
        self.D_yy = np.zeros((Nx-2,Ny-2))
        self.sigma_xx = np.zeros((Nx,Ny))
        self.sigma_xy = np.zeros((Nx,Ny))
        self.sigma_yx = np.zeros((Nx,Ny))
        self.sigma_yy = np.zeros((Nx,Ny))

        if load == False:
        # #initalise phi
            if init_type:
                self.init_phi_flat()
            else:
                self.init_phi_many()
            #initialise c
            self.init_c()
        else:
            self.phi = np.loadtxt(f'./'+name+' data/phi'+str(load_no)+'.txt')
            self.c = np.loadtxt(f'./'+name+' data/c'+str(load_no)+'.txt')
            self.px = np.loadtxt(f'./'+name+' data/px'+str(load_no)+'.txt')

    def init_phi_many(self):
        '''initalise phi as many small drops'''
        # Initialize the grid with gas (phi == -1)
        print("many")
        self.phi[:,:] = -1.0

        rows, cols = 10, 15

        # Calculate spacing between circle centers
        spacing_x = self.Nx // (cols + 1)
        spacing_y = self.Ny // (rows + 1)
        
        circle_radius = 10

        # Place the circles in a lattice pattern
        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                # Calculate the center of the circle in the lattice
                center_x = j * spacing_x
                center_y = i * spacing_y
                
                # Update the grid to set phi == 1 inside the circle
                for x in range(center_x - circle_radius, center_x + circle_radius):
                    for y in range(center_y - circle_radius, center_y + circle_radius):
                        if (x - center_x) ** 2 + (y - center_y) ** 2 <= circle_radius ** 2:
                            self.phi[x, y] = 1  # Set phi to 1 for fluid inside the circle
    

    def init_phi_flat(self):
        '''initalise phi as a flat vertical interface'''
        print("half")
        half = int(self.Nx/2)
        self.phi[:,:] = 1.0
        self.phi[:half,:] = -1.0
        
        #plotting of initial condition
        
        #plt.pcolormesh(self.X,self.Y,self.phi)
        #plt.show()

    def init_c(self):
        '''initalise c as uniform everywhere'''
        Lx = self.Lx - 2*self.dx
        Ly = self.Ly - 2*self.dx
        c_init = self.Np/(Lx*Ly)
        self.c[1:-1,1:-1] = c_init
        self.BC_standard(self.c)

        #plotting of initial condition
        
        #plt.pcolormesh(self.X,self.Y,self.c)
        #plt.show()

    def dev_x(self,q):
        return (q[2:,1:-1] - q[0:-2,1:-1]) / (2*self.dx)
    
    def dev_y(self,q):
        return (q[1:-1,2:] - q[1:-1,0:-2]) / (2*self.dx)
    
    def lap(self,q):
        return (q[2:,1:-1] + q[0:-2,1:-1] + q[1:-1,2:] + q[1:-1,0:-2]- 4*q[1:-1,1:-1]) / (self.dx*self.dx)
    
    def div(self,qx,qy):
        '''calculates divergence of a vector field (nabla . q) '''
        dq_dx = self.dev_x(qx)
        dq_dy = self.dev_y(qy)
        div = dq_dx + dq_dy

        return div

    def grad(self,q):
        '''calculates the gradient of a scalar field (nabla(q)) '''
        dq_dx = self.dev_x(q)
        dq_dy = self.dev_y(q)
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
        #grad u_x
        du_x_dx, du_x_dy = self.grad(self.u_x)
        #grad u_y
        du_y_dx, du_y_dy = self.grad(self.u_y)
        #grad_c
        dc_dx, dc_dy = self.grad(self.c)

        #PRELIMINARY CALCULATIONS FOR UPDATED VARIABLES
        #calculate mu_phi
        self.c_p_x[cs] = self.c[cs]*self.p_x[cs]
        self.c_p_y[cs] = self.c[cs]*self.p_y[cs]
        self.BC_standard(self.c_p_x)
        self.BC_standard(self.c_p_y)
        self.mu_phi[cs] = self.beta*(self.phi[cs]**3 - self.phi[cs]) -0.5*self.kappa*self.lap(self.phi) - self.xl*self.div(self.c_p_x,self.c_p_y)
        self.BC_standard(self.mu_phi)

        #calculate 2/3 of mu_c
        self.mu_c[cs] = self.kBT*(self.p_x[cs]**2 + self.p_y[cs]**2) + self.xl*(self.p_x[cs]*dphi_dx + self.p_y[cs]*dphi_dy)
        self.BC_neg_standard(self.mu_c)

        #calculate h
        self.h_x[cs] = 2*self.kBT*self.c[cs]*self.p_x[cs] + self.xl*self.c[cs]*dphi_dx
        self.h_y[cs] = 2*self.kBT*self.c[cs]*self.p_y[cs] + self.xl*self.c[cs]*dphi_dy
        self.BC_standard(self.h_x)
        self.BC_standard(self.h_y)
        
        #calculate omega
        self.omega_xy = 0.5*(du_y_dx - du_x_dy)
        self.omega_yx = 0.5*(du_x_dy - du_y_dx)

        #calculate D
        self.D_xx = du_x_dx
        self.D_xy = 0.5*(du_y_dx + du_x_dy)
        self.D_yy = du_y_dy

        #calculate sigma
        self.sigma_xx[cs] = (self.B/2) * (self.p_x[cs]*self.h_x[cs])
        self.sigma_xy[cs] = (self.p_x[cs]*self.h_y[cs])*((self.B/4) - 0.5) + (self.p_y[cs]*self.h_x[cs])*((self.B/4)+0.5)
        self.sigma_yx[cs] = (self.p_y[cs]*self.h_x[cs])*((self.B/4) - 0.5) + (self.p_x[cs]*self.h_y[cs])*((self.B/4)+0.5)
        self.sigma_yy[cs] = (self.B/2) * (self.p_y[cs]*self.h_y[cs])
        self.BC_standard(self.sigma_xx)
        self.BC_standard(self.sigma_xy)
        self.BC_standard(self.sigma_yx)
        self.BC_standard(self.sigma_yy)

        #calculate f
        dh_x_dx, dh_x_dy = self.grad(self.h_x)
        dh_y_dx, dh_y_dy = self.grad(self.h_y)
        self.f_x[cs] = -self.phi[cs]*self.dev_x(self.mu_phi) - self.c[cs]*self.dev_x(self.mu_c) - self.kBT*dc_dx - self.p_x[cs]*dh_x_dx - self.p_y[cs]*dh_y_dx + self.dev_x(self.sigma_xx) \
            + self.dev_y(self.sigma_yx)
        self.f_y[cs] = -self.phi[cs]*self.dev_y(self.mu_phi) - self.c[cs]*self.dev_y(self.mu_c) - self.kBT*dc_dy - self.p_x[cs]*dh_x_dy - self.p_y[cs]*dh_y_dy + self.dev_x(self.sigma_xy) \
            + self.dev_y(self.sigma_yy)
        self.BC_zeros(self.f_x)
        self.BC_zeros(self.f_y)

        #UPDATE VARIABLES
        #update phi
        self.phi[cs] += self.dt*(-self.u_x[cs]*dphi_dx - self.u_y[cs]*dphi_dy + self.M*self.lap(self.mu_phi)) 
        self.BC_standard(self.phi)

        #update c
        self.dmu_c_dx[cs],self.dmu_c_dy[cs] = (self.c[cs]/self.gamma_t)*self.grad(self.mu_c)
        self.BC_neg_standard(self.dmu_c_dx)
        self.BC_neg_standard(self.dmu_c_dy)
        self.c[cs] += self.dt*(-self.u_x[cs]*dc_dx - self.u_y[cs]*dc_dy + (self.kBT/self.gamma_t)*self.lap(self.c) + self.div(self.dmu_c_dx,self.dmu_c_dy))    
        self.BC_standard(self.c)   

        #update px
        self.p_x[cs] += self.dt*(-self.u_x[cs]*self.dev_x(self.p_x) - self.u_y[cs]*self.dev_y(self.p_x) + self.omega_xy*self.p_y[cs] - (self.B/2)*(self.D_xx*self.p_x[cs] + self.D_xy*self.p_y[cs])\
                                 - (1/(2*self.gamma_r*self.c[cs]))*self.h_x[cs])
        self.BC_standard(self.p_x)

        #update py
        self.p_y[cs] += self.dt*(-self.u_x[cs]*self.dev_x(self.p_y) - self.u_y[cs]*self.dev_y(self.p_y) + self.omega_yx*self.p_x[cs] - (self.B/2)*(self.D_xy*self.p_x[cs] + self.D_yy*self.p_y[cs])\
                                 - (1/(2*self.gamma_r*self.c[cs]))*self.h_y[cs])
        self.BC_standard(self.p_y)

        #create flipped f
        fx = np.zeros((2*self.Nx,self.Ny))
        fx[:self.Nx,:] = self.f_x
        fx[self.Nx:,:] = -self.f_x[::-1,:]

        fy = np.zeros((2*self.Nx,self.Ny))
        fy[:self.Nx,:] = self.f_y
        fy[self.Nx:,:] = -self.f_y[::-1,:]
        
        #fourier parameters
        fx_k = np.fft.fft2(fx,norm='ortho')*self.dx
        fy_k = np.fft.fft2(fy,norm='ortho')*self.dx

        #preliminary calculations for velocity
        k_f = (self.KX*fx_k) + (self.KY*fy_k)
    
        #to avoid a divide by 0
        ind = (self.k2 != 0)
        #index that is dodged remains zeros anyways (i think)
    
        #velocity calculations
        self.u_x_k[ind] = ((fx_k[ind]/self.k2[ind]) - (self.KX[ind]*k_f[ind]/self.k4[ind]))/self.eta
        self.u_y_k[ind] = ((fy_k[ind]/self.k2[ind]) - (self.KY[ind]*k_f[ind]/self.k4[ind]))/self.eta
    
        #transform u back into real space
        self.u_x = (np.fft.ifft2(self.u_x_k,norm='ortho').real/self.dx)[:self.Nx,:]
        self.u_y = (np.fft.ifft2(self.u_y_k,norm='ortho').real/self.dx)[:self.Nx,:]
        
    def run(self):
        '''running loop'''
        #indexes [1:-1,1:-1], to keep from rewriting it and taking up space
        #cs = central slice
        cs = (slice(1,-1),slice(1,-1))
        phi_con = 0
        c_con = self.Np

        #run entire sim
        for i in range(self.Nt + 1):
            self.update_all(cs)

            # #print conservations
            # phi_sum = np.sum(self.phi[1:-1,1:-1]*self.dx*self.dx)
            # c_sum = np.sum(self.c[1:-1,1:-1]*self.dx*self.dx)
            # if np.round(phi_sum-phi_con) != 0.0:
            #     print("phi not conserved")
            # if np.round(c_sum-c_con) != 0.0:
            #     print("c not conserved")
            #print(f'phi sum: {phi_sum}   c sum: {c_sum}')

            #save files
            if i % self.save == 0:
                np.savetxt(f'./'+self.name+' data/phi'+str(i+self.load_no)+'.txt',self.phi)
                np.savetxt(f'./'+self.name+' data/c'+str(i+self.load_no)+'.txt',self.c)
                np.savetxt(f'./'+self.name+' data/px'+str(i+self.load_no)+'.txt',self.p_x)
                np.savetxt(f'./'+self.name+'data/py'+str(i+self.load_no)+'.txt',self.p_y)


#intiate class
try:
    s = Surfactant(name='ultrastable emulsion',xl=0.1,init_type=0, dt = 0.001)
    if s.crash == False:
        s.run()
except:
    pass

