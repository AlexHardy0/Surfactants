# imports
import numpy as np
import os
import gc
import cProfile
import pstats
import traceback


#############################################################################################################################################################
# CLASSES USED FOR RUNNING
# all simulations are contained in their own classes
# Surfactant is the class used for surfactant runs
# Surfactant_alternative is the surfactant-less version
#############################################################################################################################################################

profiler = cProfile.Profile()
profiler.enable()

# SURFACTANT_ALTERNATIVE CLASS
class Surfactant_alternative:
    def __init__(self,
                 dt = 0.001, dx = 0.5, Nx = 64, Ny = 64, Nt = int(1e6), 
                 M = 3.0, beta = 2.0, kappa = 1.0,  eta = 1.0, 
                 name = 'new dx_0.5'):
        ''' 
        
        Initalise every variable/array to be used in the simulation as a property
        
        Parameters: all variables which can be set in the simulation, automatically set to their default value if none are specified
        they are split as:
        line 1: resolution parameters, time step, spatial res., size of x-direction, size of y-direction, Total No. of time steps
        line 2: equation parameters: M, beta, kappa and eta
        line 3: name of the simulation folder to save to
        
        '''
        # set up folder to save data to
        self.name = name
        self.crash = False

        try:
            os.mkdir('./'+name+' data')
        except OSError:
            print("folder name in use")
            # stop the simulation to avoid overwriting data
            self.crash = True

        # set up time parameters
        self.dt = dt
        self.Nt = Nt
        self.save = int(Nt / 100)

        # set up space parameters
        self.dx = dx  #assuming dx == dy
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = int(Nx*dx)
        self.Ly = int(Ny*dx)

        # meshgrid for plotting
        x = np.arange(0.0,self.Lx,self.dx)
        y = np.arange(0.0,self.Ly,self.dx)
        self.Y,self.X = np.meshgrid(y,x)

        # set up fourier space parameters
        kx = np.fft.fftfreq(Nx,d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny,d=dx) * 2 * np.pi
        self.KY,self.KX = np.meshgrid(ky,kx)
        self.k2 = self.KX**2 + self.KY**2
        self.k4 = self.k2 * self.k2

        # set up constants
        self.M = M
        self.beta = beta
        self.kappa = kappa
        self.eta = eta

        # scalar field array definitions
        self.phi = np.zeros((Nx,Ny))
        self.mu_phi = np.zeros((Nx,Ny))

        # vector field array definitions
        self.u_x = np.zeros((Nx,Ny))
        self.u_y = np.zeros((Nx,Ny))
        self.u_x_k = np.zeros((Nx,Ny),dtype = complex)
        self.u_y_k = np.zeros((Nx,Ny),dtype = complex)
        self.f_x = np.zeros((Nx,Ny))
        self.f_y = np.zeros((Nx,Ny))

        # initalise phi - currently set up to do ultrastable emulsions, for flat interfaces, use the init_flat() function instead
        self.init_phi_many()
        # self.init_flat()

    def init_flat(self):
        '''
        Initalizes the phi array as two halves of fluid, with a flat vertical interface halfway 
        '''

        print("half")
        half = int(self.Nx/2)
        self.phi[:,:] = 1.0
        self.phi[:half,:] = -1.0

    def init_phi_many(self):
            '''
            Initialize phi with multiple small drops
            '''
            # Initialize the grid with gas 
            print("many")
            self.phi[:, :] = -1.0

            # small drop parameters
            size = self.Nx - 2
            center = size // 2
            off_center = size // 4
            radius = 9

            # function to place a circle in phi
            def place_circle(xc, yc, r):
                for x in range(size):
                    for y in range(size):
                        if (x - xc) ** 2 + (y - yc) ** 2 < r ** 2:
                            self.phi[x+1, y+1] = 1.0

            # place the corner drop
            corner_positions = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
            for (xc, yc) in corner_positions:
                place_circle(xc, yc, radius)

            # place the top and bottom drops
            place_circle(center, 0, radius)       
            place_circle(center, size - 1, radius) 

            # place the left and right drops
            place_circle(0, center, radius,)      
            place_circle(size - 1, center, radius)   

            # place drop in the centre
            place_circle(center, center, radius)

            # place drops in the gaps between
            place_circle(off_center,off_center,radius)
            place_circle(off_center+center,off_center,radius)
            place_circle(off_center,off_center+center,radius)
            place_circle(off_center+center,off_center+center,radius)

            # apply boundary conditions
            self.BC_standard(self.phi)

            # Uncomment the following lines if you want to visualize the result
            # plt.pcolormesh(self.X, self.Y, self.phi)
            # plt.show()

    def dev_x(self,q):
        '''derivative in x direction, q: array to differentiate'''
        return (q[2:,1:-1] - q[0:-2,1:-1]) / (2*self.dx)
    
    def dev_y(self,q):
        '''derivative in y direction, q: array to differentiate'''
        return (q[1:-1,2:] - q[1:-1,0:-2]) / (2*self.dx)
    
    def grad(self,q):
        '''calculates the gradient of a scalar field (grad(q))'''
        dq_dx = self.dev_x(q)
        dq_dy = self.dev_y(q)
        grad = [dq_dx,dq_dy]

        return grad
    
    def lap(self,q):
        '''calculates grad^2(q)'''
        return (q[2:,1:-1] + q[0:-2,1:-1] + q[1:-1,2:] + q[1:-1,0:-2]- 4*q[1:-1,1:-1]) / (self.dx*self.dx)
    
    def BC_standard(self,q):
        '''apply standard boundary conditions to array q'''
        #standard = periodic on all, will need to be changed if performing flat interface sims
        q[0,:] = q[-2,:]
        q[-1,:] = q[1,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def BC_zeros(self,q):
        '''apply current boundary conditions to array q'''
        #standard = periodic on all, will need to be changed if performing flat interface sims
        #typically would set wall values of q to zero (no slip BC)
        q[0,:] = q[-2,:]
        q[-1,:] = q[1,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def update_all(self,cs):
        '''
        Updates each variable used for one time step

        Parameters: cs = centre slice, used to only update the 'real' values for each field
        '''
        # CALCULATE COMMONLY USED GRADIENTS
        # grad_phi
        dphi_dx,dphi_dy = self.grad(self.phi)
        
        #######################################################################################################################################################
        #PRELIMINARY CALCULATIONS FOR UPDATED VARIABLES + APPLY APPROPIATE BC'S
        #######################################################################################################################################################
        # calculate mu_phi
        self.mu_phi[cs] = self.beta*(self.phi[cs]**3 - self.phi[cs]) -self.kappa*self.lap(self.phi)
        self.BC_standard(self.mu_phi)
        #######################################################################################################################################################
        # calculate f
        self.f_x[cs] = -self.phi[cs]*self.dev_x(self.mu_phi) 
        self.f_y[cs] = -self.phi[cs]*self.dev_y(self.mu_phi) 
        self.BC_zeros(self.f_x)
        self.BC_zeros(self.f_y)

        #######################################################################################################################################################
        # UPDATE ALL FIELDS + APPLY APPROPIATE BC'S
        #######################################################################################################################################################
        # update phi
        self.phi[cs] += self.dt*(-self.u_x[cs]*dphi_dx - self.u_y[cs]*dphi_dy + self.M*self.lap(self.mu_phi)) 
        self.BC_standard(self.phi)
        #######################################################################################################################################################
        # create flipped f - only used for the flat interface sims
        # fx = np.zeros((self.Nx,self.Ny))
        # fx[:self.Nx,:] = self.f_x
        # fx[self.Nx:,:] = -self.f_x[::-1,:]

        # fy = np.zeros((self.Nx,self.Ny))
        # fy[:self.Nx,:] = self.f_y
        # fy[self.Nx:,:] = -self.f_y[::-1,:]
        
        fx = self.f_x
        fy = self.f_y
        #######################################################################################################################################################
        # fourier parameters
        fx_k = np.fft.fft2(fx,norm='ortho')*self.dx
        fy_k = np.fft.fft2(fy,norm='ortho')*self.dx

        # preliminary calculations for velocity
        k_f = (self.KX*fx_k) + (self.KY*fy_k)
    
        # to avoid a divide by 0
        ind = (self.k2 != 0)
        # index that is dodged remains zeros anyways (i think)
        #######################################################################################################################################################
        # velocity calculations
        self.u_x_k[ind] = ((fx_k[ind]/self.k2[ind]) - (self.KX[ind]*k_f[ind]/self.k4[ind]))/self.eta
        self.u_y_k[ind] = ((fy_k[ind]/self.k2[ind]) - (self.KY[ind]*k_f[ind]/self.k4[ind]))/self.eta
    
        # transform u back into real space
        self.u_x = (np.fft.ifft2(self.u_x_k,norm='ortho').real/self.dx)[:self.Nx,:]
        self.u_y = (np.fft.ifft2(self.u_y_k,norm='ortho').real/self.dx)[:self.Nx,:]
        
    def run(self):
        '''
        running loop for entire simulation
        '''
        #indexes [1:-1,1:-1], to keep from rewriting it and taking up space
        #cs = central slice
        cs = (slice(1,-1),slice(1,-1))

        #run entire sim
        for i in range(self.Nt + 1):
            self.update_all(cs)

            #save files
            if i % self.save == 0:
                print(np.sum(self.phi[cs]*self.dx*self.dx))  #conservation check
                np.savetxt(f'./'+self.name+' data/phi'+str(i)+'.txt',self.phi)
                
# SURFACTANT CLASS
class Surfactant:
    def __init__(self,
                 dt = 0.01, dx = 0.5, Nx = 512, Ny = 256, Nt = int(1e6), 
                 M = 3.0, beta = 2.0, kappa = 1.0, xl = 0.1, kBT = 1.0,
                 B = 0.5, gamma_r = 0.01, gamma_t = 0.01, eta = 1.0, Np = 500,
                 name = 'new dx_0.5',load=False, load_no = 0, init_type = 1):
        ''' 
        
        Initalise every variable/array to be used in the simulation as a property
        
        Parameters: all variables which can be set in the simulation, automatically set to their default value if none are specified
        they are split as:
        line 1: resolution parameters, time step, spatial res., size of x-direction, size of y-direction, Total No. of time steps
        line 2: equation parameters: M, beta, kappa, xl, kBT
        line 3: continued equation parameters: B, gamma_r, gamma_t, eta and Np, the number of particles to use
        line 4: name of the simulation folder to save to, loading information, and type of simulation run
        
        '''
        #set up folder to save data to
        self.name = name
        self.type = init_type
        self.crash = False
        self.load_no = load_no

        try:
            os.mkdir('./'+name+'_data')
        except OSError:
            print("folder name in use")
            if load == False:
                # stop the simulation to avoid overwriting data, except if you are loading from that folder
                # self.crash = True
                pass

        # set up Np
        self.Np = Np

        # set up time parameters
        self.dt = dt
        self.Nt = Nt
        self.save = int(Nt / 100)

        # set up space parameters
        self.dx = dx  #assuming dx == dy
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = int(Nx*dx)
        self.Ly = int(Ny*dx)

        # meshgrid for plotting
        x = np.arange(0.0,self.Lx,self.dx)
        y = np.arange(0.0,self.Ly,self.dx)
        self.Y,self.X = np.meshgrid(y,x)

        # set up fourier space parameters, based on if walls are present or not
        if self.type:
            kx = np.fft.fftfreq(2*Nx,d=dx) * 2 * np.pi
        else:
            kx = np.fft.fftfreq(Nx,d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny,d=dx) * 2 * np.pi
        kx[0] = 0.001
        ky[0] = 0.001
        KY,KX = np.meshgrid(ky,kx)
        self. K = np.stack((KX,KY))
        self.k2 = self.K[0]*self.K[0] + self.K[1]*self.K[1]
        self.k4 = self.k2 * self.k2

        # set up constants
        self.M = M
        self.beta = beta
        self.kappa = kappa
        self.xl = xl
        self.B = B
        self.gamma_t = gamma_t
        self.gamma_r = gamma_r
        self.eta = eta
        self.kBT = kBT

        # scalar field array definitions
        self.phi = np.zeros((Nx,Ny),dtype='float32')
        self.mu_phi = np.zeros((Nx,Ny),dtype='float32')
        self.c = np.zeros((Nx,Ny),dtype='float32')
        self.mu_c = np.zeros((Nx,Ny),dtype='float32')

        # vector field array definitions
        self.u = np.zeros((2,Nx,Ny),dtype='float32')
        if self.type:
            self.u_k = np.stack((np.zeros((2*Nx,Ny),dtype = complex),np.zeros((2*Nx,Ny),dtype = complex)))
        else:
            self.u_k = np.stack((np.zeros((Nx,Ny),dtype = complex),np.zeros((Nx,Ny),dtype = complex)))
        self.p = np.zeros((2,Nx,Ny),dtype='float32')
        self.f = np.zeros((2,Nx,Ny),dtype='float32')
        self.h = np.zeros((2,Nx,Ny),dtype='float32')
        self.J = np.zeros((2,Nx,Ny),dtype='float32')

        #derivatives field array definitions
        self.dphi = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.dc = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.du_x = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.du_y = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.dp_x = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.dp_y = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.dmu_c = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.dmu_phi = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.dh_dxs = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.dh_dys = np.zeros((2,Nx-2,Ny-2),dtype='float32')

        #combinations field array definitions
        self.c_p = np.zeros((2,Nx,Ny),dtype='float32')
        self.p_dh = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.dsigma = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.u_dp = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        self.p_D = np.zeros((2,Nx-2,Ny-2),dtype='float32')

        # matrix array defintions
        # omega_xx amd omega_yy = 0
        #omega xy, omega yx
        self.omega = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        #D_xx D_xy
        self.D_x = np.zeros((2,Nx-2,Ny-2),dtype='float32')
        #D_xy D_yy
        self.D_y = np.zeros((2,Nx-2,Ny-2),dtype='float32')

        self.sigma_xx = np.zeros((Nx,Ny),dtype='float32')
        self.sigma_xy = np.zeros((Nx,Ny),dtype='float32')
        self.sigma_yx = np.zeros((Nx,Ny),dtype='float32')
        self.sigma_yy = np.zeros((Nx,Ny),dtype='float32')

        # intialize phi and c, based on simulation type + if you are loading 
        if load == False:
            # initalise phi
            if self.type:
                self.init_phi_flat()
            else:
                self.init_phi_many()
            #initialise c
            self.init_c()
        else:
            #load from folder
            self.phi = np.loadtxt(f'./'+name+' data/phi'+str(load_no)+'.txt')
            self.c = np.loadtxt(f'./'+name+' data/c'+str(load_no)+'.txt')
            self.px = np.loadtxt(f'./'+name+' data/px'+str(load_no)+'.txt')
            self.py = np.loadtxt(f'./'+name+' data/py'+str(load_no)+'.txt')

    def init_phi_many(self):
        '''
        Initialize phi with multiple small drops
        '''
        # Initialize the grid with gas
        print("many")
        self.phi[:, :] = -1.0

        # small drop parameters
        size = self.Nx - 2
        center = size // 2
        off_center = size // 4
        radius = 9

        # Function to place a circle in phi
        def place_circle(xc, yc, r):
            '''
            function for placing small drops in phi

            Parameters:
            xc : center x coord
            yc : center y coord
            r  : radius of drop
            '''
            for x in range(size):
                for y in range(size):
                    if (x - xc) ** 2 + (y - yc) ** 2 < r ** 2:
                        self.phi[x+1, y+1] = 1.0

        # Place the corner drop
        corner_positions = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
        for (xc, yc) in corner_positions:
            place_circle(xc, yc, radius)

        # Place the top and bottom drops
        place_circle(center, 0, radius)       
        place_circle(center, size - 1, radius) 

        # Place the left and right drops
        place_circle(0, center, radius,)      
        place_circle(size - 1, center, radius)   

        # Place drop in the centre
        place_circle(center, center, radius)

        # Place dropin the gaps between
        place_circle(off_center,off_center,radius)
        place_circle(off_center+center,off_center,radius)
        place_circle(off_center,off_center+center,radius)
        place_circle(off_center+center,off_center+center,radius)

        # apply boundary conditions
        self.BC_standard(self.phi)
        
        # Uncomment the following lines if you want to visualize the result
        # plt.pcolormesh(self.X, self.Y, self.phi)
        # plt.show()

    def init_phi_flat(self):
        '''
        Initalizes the phi array as two halves of fluid, with a flat vertical interface halfway 
        '''
        print("half")
        half = int(self.Nx/2)
        self.phi[:,:] = 1.0
        self.phi[:half,:] = -1.0
        
        # Uncomment the following lines if you want to visualize the result
        
        #plt.pcolormesh(self.X,self.Y,self.phi)
        #plt.show()

    def init_c(self):
        '''
        Initalise c as uniform everywhere
        '''
        Lx = self.Lx - 2*self.dx
        Ly = self.Ly - 2*self.dx
        c_init = self.Np/(Lx*Ly)
        self.c[1:-1,1:-1] = c_init

        #aplpy boundary conditions
        self.BC_standard(self.c)

        # Uncomment the following lines if you want to visualize the result
        #plt.pcolormesh(self.X,self.Y,self.c)
        #plt.show()

    def dev_x(self,q):
        '''derivative in x direction, q: array to differentiate'''
        dev = (q[2:,1:-1] - q[0:-2,1:-1]) / (2*self.dx)
        return dev
    
    def dev_y(self,q):
        '''derivative in y direction, q: array to differentiate'''
        dev = (q[1:-1,2:] - q[1:-1,0:-2]) / (2*self.dx)
        return dev
    
    def lap(self,q):
        '''calculates grad^2(q)'''
        lap = (q[2:,1:-1] + q[0:-2,1:-1] + q[1:-1,2:] + q[1:-1,0:-2]- 4*q[1:-1,1:-1]) / (self.dx*self.dx)
        return lap
    
    def div(self,qx,qy):
        '''calculates divergence of a vector field (nabla . q) '''
        div = (qx[2:,1:-1] - qx[0:-2,1:-1] + qy[1:-1,2:] - qy[1:-1,0:-2]) / (2*self.dx)

        return div
    
    def dot(self,q1,q2):
        '''calculates dot product from STACKS (q1 and q2)'''
        dot = q1[0]*q2[0] + q1[1]*q2[1]
        return dot
    
    def neg_dot(self,q1,q2):
        '''calculates dot product from STACKS (q1 and q2) except instead of adding it subtracts'''
        dot = q1[0]*q2[0] - q1[1]*q2[1]
        return dot

    def BC_standard(self,q):
        '''apply standard boundary conditions to array q'''
        #standard = periodic on y walls, neumann on x walls for flat interface
        #standard = periodic on all walls for emulsion
        if self.type:
            q[0,:] = q[1,:]
            q[-1,:] = q[-2,:]
        else:
            q[0,:] = q[-2,:]
            q[-1,:] = q[1,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def BC_neg_standard(self,q):
        '''apply negative standard boundary conditions to array q'''
        #standard = periodic on y walls, negative neumann on x walls for flat interface
        #standard = periodic on all walls for emulsion
        if self.type:
            q[0,:] = -q[1,:]
            q[-1,:] = -q[-2,:]
        else:
            q[0,:] = q[-2,:]
            q[-1,:] = q[1,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def BC_zeros(self,q):
        '''apply zero boundary conditions to array q'''
        #standard = periodic on y walls, zero on x walls for flat interface
        #standard = periodic on all walls for emulsion
        if self.type:
            q[0,:] = 0
            q[-1,:] = 0
        else:
            q[0,:] = q[-2,:]
            q[-1,:] = q[1,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def calc_mu_phi(self,cs):
         #calculate mu_phi
        self.c_p = self.c*self.p
        self.BC_standard(self.c_p[0])
        self.BC_standard(self.c_p[1])
        
        self.mu_phi[cs] = self.beta*(self.phi[cs]*self.phi[cs]*self.phi[cs] - self.phi[cs]) - self.kappa*self.lap(self.phi)\
              - self.xl*self.div(self.c_p[0],self.c_p[1])
        self.BC_standard(self.mu_phi)

    def calc_mu_c(self,cs,css):
        # calculate 2/3 of mu_c
        self.mu_c[cs] = self.kBT*(self.dot(self.p[css],self.p[css])) + self.xl*(self.dot(self.p[css],self.dphi))
        self.BC_standard(self.mu_c)

        self.dmu_c[0],self.dmu_c[1] = self.dev_x(self.mu_c),self.dev_y(self.mu_c)

    def calc_h(self,css):
        # calculate h (without c)
        self.h[css] = 2*self.kBT*self.p[css] + self.xl*self.dphi
        self.BC_standard(self.h[0])
        self.BC_standard(self.h[1])

    def calc_omega(self):
        # calculate omega
        self.omega[0] = 0.5*(self.du_y[0] - self.du_x[1])
        self.omega[1] = 0.5*(self.du_x[1] - self.du_y[0])

    def calc_D(self):
        # calculate D
        self.D_x[0] = self.du_x[0]
        self.D_x[1] = 0.5*(self.du_y[0] + self.du_x[1])
        self.D_y[0] = 0.5*(self.du_y[0] + self.du_x[1])
        self.D_y[1] = self.du_y[1]

    def calc_sigma(self,cs,css,rcss,c_h):
        # calculate sigma
        p_ch = self.p*c_h
    
        self.sigma_xx[cs],self.sigma_yy[cs] = (0.5*self.B) * p_ch[css]
        self.sigma_xy[cs] = ((0.25*self.B)*self.dot(self.p[css],c_h[rcss])) - 0.5*self.neg_dot(self.p[css],c_h[rcss])
        self.sigma_yx[cs] = ((0.25*self.B)*self.dot(self.p[rcss],c_h[css])) - 0.5*self.neg_dot(self.p[rcss],c_h[css])
        
        self.BC_standard(self.sigma_xx)
        self.BC_standard(self.sigma_xy)
        self.BC_standard(self.sigma_yx)
        self.BC_standard(self.sigma_yy)

    def calc_f(self,cs,css):
        # calculate f
        self.dmu_phi[0] = self.dev_x(self.mu_phi)
        self.dmu_phi[1] = self.dev_y(self.mu_phi)
        self.p_dh[0] = self.dot(self.p[css],self.dh_dxs)
        self.p_dh[1] = self.dot(self.p[css],self.dh_dys)
        self.dsigma[0] = self.div(self.sigma_xx,self.sigma_yx)
        self.dsigma[1] = self.div(self.sigma_xy,self.sigma_yy)

        self.f[css] = -self.phi[cs]*self.dmu_phi - self.c[cs]*self.dmu_c - self.kBT*self.dc - self.p_dh + self.dsigma

        self.BC_zeros(self.f[0])
        self.BC_zeros(self.f[1])

    def update_phi(self,cs,css):
        # update phi
        self.phi[cs] += self.dt*(-self.dot(self.u[css],self.dphi)+ self.M*self.lap(self.mu_phi)) 
        self.BC_standard(self.phi)

    def update_c(self,cs,css):
        # update c
        self.J[css] = (self.c[cs]/self.gamma_t)*self.dmu_c
        self.BC_neg_standard(self.J[0])
        self.BC_neg_standard(self.J[1])
        
        self.c[cs] += self.dt*(-self.dot(self.u[css],self.dc) + (self.kBT/self.gamma_t)*self.lap(self.c) + self.div(self.J[0],self.J[1]))    
        self.BC_standard(self.c)   

    def update_p(self,css,rcss):
        self.dp_x[0] = self.dev_x(self.p[0])
        self.dp_x[1] = self.dev_y(self.p[0])
        self.dp_y[0] = self.dev_x(self.p[1])
        self.dp_y[1] = self.dev_y(self.p[1])
        self.u_dp[0] = self.dot(self.u[css],self.dp_x)
        self.u_dp[1] = self.dot(self.u[css],self.dp_y)
        self.p_D[0] = self.dot(self.p[css],self.D_x)
        self.p_D[1] = self.dot(self.p[css],self.D_y)

        self.p[css] += self.dt*(-self.u_dp + self.omega*self.p[rcss] - (0.5*self.B)*self.p_D - 0.5*(1/self.gamma_r)*self.h[css])
        
        self.BC_standard(self.p[0])
        self.BC_standard(self.p[1])

    def update_u(self):
        #create flipped f (if walls  are on)
        if self.type:
            f = np.stack((np.zeros((2*self.Nx,self.Ny)),np.zeros((2*self.Nx,self.Ny))))
            f[0,:self.Nx,:] = self.f[0]
            f[0,self.Nx:,:] = -self.f[0,::-1,:]

            f[1,:self.Nx,:] = self.f[1]
            f[1,self.Nx:,:] = -self.f[1,::-1,:]
        else:
            f = self.f
        #######################################################################################################################################################
        # fourier parameters
        f_k = np.fft.fft2(f,norm='ortho')*self.dx

        # preliminary calculations for velocity
        k_f = f_k[0]*self.K[0] + f_k[1]*self.K[1]
    
        # index that is dodged remains zeros anyways (i think)
        ######################################################################################################################################################
        # velocity calculations
        self.u_k = ((f_k/self.k2) - (self.K*k_f/self.k4))/self.eta
    
        # transform u back into real space
        self.u = (np.fft.ifft2(self.u_k,norm='ortho').real/self.dx)[:,:self.Nx,:]
    
    def update_all(self,cs,css,rcss):
        '''
        Updates each variable used for one time step

        Parameters: cs = centre slice, used to only update the 'real' values for each field
        '''
        # CALCULATE COMMONLY USED GRADIENTS
        # grad_phi
        self.dphi[0] = self.dev_x(self.phi)
        self.dphi[1] = self.dev_y(self.phi)
        # grad_c
        self.dc[0] = self.dev_x(self.c)
        self.dc[1] = self.dev_y(self.c)
        # grad u_x
        self.du_x[0] = self.dev_x(self.u[0])
        self.du_x[1] = self.dev_y(self.u[0])
        # grad u_y
        self.du_y[0] = self.dev_x(self.u[1])
        self.du_y[1] = self.dev_y(self.u[1])
        
        #######################################################################################################################################################
        #PRELIMINARY CALCULATIONS FOR UPDATED VARIABLES + APPLY APPROPIATE BC'S
        #######################################################################################################################################################
        self.calc_mu_phi(cs)
        #######################################################################################################################################################
        self.calc_mu_c(cs,css)
        #######################################################################################################################################################
        self.calc_h(css)
        #######################################################################################################################################################
        self.calc_omega()
        #######################################################################################################################################################
        self.calc_D()
        #######################################################################################################################################################
        c_h = self.c * self.h
        self.BC_standard(c_h[0])
        self.BC_standard(c_h[1])

        self.calc_sigma(cs,css,rcss,c_h)
        #######################################################################################################################################################
        # dchx dx  dchy dx
        self.dh_dxs[0] = self.dev_x(c_h[0]) 
        self.dh_dxs[1] = self.dev_x(c_h[1])
        # dchx dy dchy dy 
        self.dh_dys[0] = self.dev_y(c_h[0])
        self.dh_dys[1] = self.dev_y(c_h[1])

        self.calc_f(cs,css)

        #######################################################################################################################################################
        # UPDATE ALL FIELDS + APPLY APPROPIATE BC'S
        #######################################################################################################################################################
        self.update_phi(cs,css)
        #######################################################################################################################################################
        self.update_c(cs,css)
        #######################################################################################################################################################
        self.update_p(css,rcss)
        #######################################################################################################################################################
        self.update_u()
        
    def run(self):
        '''
        running loop for entire simulation
        '''

        #indexes [1:-1,1:-1], to keep from rewriting it and taking up space
        #cs = central slice
        cs = (slice(1,-1),slice(1,-1))
        #css = central slice stack
        css = (slice(None),slice(1,-1),slice(1,-1))
         #reverse css
        rcss = (slice(-1,None,-1),slice(1,-1),slice(1,-1))

        #run entire sim
        for i in range(self.Nt + 1):
            self.update_all(cs,css,rcss)

            #save files, continues to use the same folder when loading
            if i % self.save == 0:
                # uncomment for conservation check
                # print(np.sum(self.c[cs]*self.dx*self.dx))  #conservation check
                # print(np.sum(self.phi[cs]*self.dx*self.dx))  #conservation check
                np.savez(f'./{self.name}_data/step_{i+self.load_no}.npz', phi=self.phi, c=self.c, px=self.p[0], py=self.p[1])


def run_simulation(simulation_class, *args, **kwargs):
    """
    Runs a simulation, handles exceptions, and frees memory afterward.

    Parameters:
    simulation_class: Class of the simulation
    args: Positional arguments for the simulation class.
    kwargs: Keyword arguments for the simulation class.
    """

    try:
        sim = simulation_class(*args, **kwargs)
        #only run when given the all clear to run
        #this is calculated in the initalization of class
        if not sim.crash:  
            sim.run()  
        print(f"{kwargs.get('name', 'Simulation')} passed")
    except Exception as e:
        tb_str = traceback.format_exc()
    
        # Print the custom message along with the traceback
        print(f"{kwargs.get('name', 'Simulation')} crashed: {e}\n{tb_str}")
    finally:
        # Explicitly delete the object and force garbage collection
        # this saves memory when running multiple simulations one after each other
        del sim
        gc.collect()

# Running the simulations, uncomment to run or make new sim

# speed test
run_simulation(Surfactant,name='speed test',Nt = int(1e3), Nx = 500, Ny = 500, dx = 1.0, dt = 0.001, init_type=1)

profiler.disable()
# Save the results to a file
with open("profile_output.txt", "w") as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats("cumulative")  # Sort by cumulative time or "time" to sort by time spent in the function
    stats.print_stats()

# tanh test
# run_simulation(Surfactant_alternative,name='tanh test',Nt=int(1e5))

# flat interface investigations 
# run_simulation(Surfactant,name='epsilon=0.9', dx = 0.25, Np = 400, dt = 0.0001, xl = 0.9 , Nt = int(5e5),load=True,load_no = int(5.5e5))
# run_simulation(Surfactant,name='epsilon=0.7', dx = 0.25, Np = 400, dt = 0.0001, xl = 0.7 , Nt = int(5e5),load=True,load_no = int(5.5e5))
# run_simulation(Surfactant,name='epsilon=0.5', dx = 0.25, Np = 400, dt = 0.0001, xl = 0.5 , Nt = int(5e5),load=True,load_no = int(5.5e5))
# run_simulation(Surfactant,name='epsilon=0.3', dx = 0.25, Np = 400, dt = 0.0001, xl = 0.3 , Nt = int(5e5),load=True,load_no = int(5.5e5))
# run_simulation(Surfactant,name='epsilon=0.1', dx = 0.25, Np = 400, dt = 0.0001, xl = 0.1 , Nt = int(5e5),load=True,load_no = int(5.5e5))
# run_simulation(Surfactant,name='epsilon=0.01',dx = 0.25, Np = 400, dt = 0.0001, xl = 0.01, Nt = int(5e5),load=True,load_no = int(5.5e5))

# ultrastable emsulsions
# run_simulation(Surfactant_alternative, name='ultrastable emulsion - no surfactant vrs', dx = 1.0, Nx = 64, Ny = 64, dt = 0.001,Nt=int(0.5e6))

# run_simulation(Surfactant, name='ultrastable emulsion 0.5 - 500',  xl=0.5, Np = 500,  init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 0.5 - 1000', xl=0.5, Np = 1000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 0.5 - 1500', xl=0.5, Np = 1500, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))

# run_simulation(Surfactant, name='ultrastable emulsion 1.0 - 500',  xl=1.0, Np = 500,  init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 1.0 - 1000', xl=1.0, Np = 1000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 1.0 - 1500', xl=1.0, Np = 1500, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))

# run_simulation(Surfactant, name='ultrastable emulsion 1.5 - 500',  xl=1.5, Np = 500,  init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 1.5 - 1000', xl=1.5, Np = 1000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 1.5 - 1500', xl=1.5, Np = 1500, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))

# run_simulation(Surfactant, name='ultrastable emulsion 2.0 - 500',  xl=2.0, Np = 500,  init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 2.0 - 1000', xl=2.0, Np = 1000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 2.0 - 1500', xl=2.0, Np = 1500, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
