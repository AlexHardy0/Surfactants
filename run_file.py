#imports
import numpy as np
import matplotlib.pyplot as plt
import os
import gc


#class definition for structure
class Droplet:
    def __init__(self,
                 dt = 0.01, dx = 1.0, Nx = 500, Ny = 250, Nt = int(0.5e6), start = int(1e2),
                 M = 1.0, beta = 1.0, kappa = 1.0, eta = 1.0, rho = 1.0, radius = 50, mp = 11,
                 name = 'test'):
        ''' initalise every variable/array '''
        #set up folder to save data to
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.realpath(__file__))

        # Define your file path relative to the script's location
        self.file_path = os.path.join(script_dir, name)
        self.name = name
        self.crash = False
        try:
            os.mkdir(self.file_path)
        except OSError:
            print("folder name in use")
            self.crash = True

        #time
        self.dt = dt
        self.start = start
        self.Nt = Nt
        self.save = int(Nt / 100)
        self.nt_pcl = 30

        #regular space
        self.dx = dx  #assuming dx == dy
        self.Nx = Nx
        self.Ny = Ny
        self.Lx = int(Nx*dx)
        self.Ly = int(Ny*dx)

        # meshgrid for plotting
        xm = self.Lx/2
        x = np.arange(-xm,xm,self.dx)
        ym = self.Ly/2
        y = np.arange(-ym,ym,self.dx)
        self.X,self.Y = np.meshgrid(x,y)
        #array size to base everything else on
        self.size = self.X.shape

        #constants
        self.M = M
        self.beta = beta
        self.kappa = kappa
        self.eta = eta
        self.rho = rho
        self.radius = radius
        self.mp = mp

        #if evaporation is on
        self.evap = False

        #drop edge
        self.l_intrfc = int((self.Lx/2 - self.radius)/self.dx)
        self.r_intrfc = int((self.Lx/2 + self.radius)/self.dx)

        #scalar field array definitions
        self.phi = np.zeros(self.size)
        self.mu_phi = np.zeros(self.size)

        #vector field array definitions
        self.ux = np.zeros(self.size)
        self.uy = np.zeros(self.size)
        self.ux_star = np.zeros(self.size)
        self.uy_star = np.zeros(self.size)
        self.p = np.zeros(self.size)

        #initalise phi
        self.init_phi()

    def init_phi(self):
        '''initalise phi as a 2d droplet'''
        #Initialisation of droplet
        circle = np.zeros(self.size)
        indy, indx = np.where(circle == 0)
        centre = ((self.Lx-self.dx)/2)
        circle[indy,indx]= (indx*self.dx - centre)**2 + (indy*self.dx)**2
        drop = (circle < self.radius**2)

        #evaporation radius
        evap_rad = (self.Ly-self.dx)/2
        self.edge = (circle > evap_rad**2)

        self.phi[:,:] = -1.0
        self.phi[drop] = 1.0

        # # checking initial symmetry
        # h = int(self.Nx/2)
        # sym = self.phi[:,:h] - np.flip(self.phi[:,h:],axis=1)
        # plt.plot(sym)
        # plt.show()
        # plotting of initial condition
        # plt.pcolormesh(self.X,self.Y,self.phi)
        # plt.show()
    
    def dev_x(self,q):
        return (q[1:-1,2:] - q[1:-1,0:-2]) / (2*self.dx)
    
    def dev_y(self,q):
        return (q[2:,1:-1] - q[0:-2,1:-1]) / (2*self.dx)
    
    def lap(self,q):
        return (q[2:,1:-1] + q[0:-2,1:-1] + q[1:-1,2:] + q[1:-1,0:-2]- 4*q[1:-1,1:-1]) / (self.dx*self.dx)

    def BC_standard(self,q):
        '''apply standard boundary conditions'''
        #standard = all walls
        q[0,:] = q[1,:]
        q[-1,:] = q[-2,:]
        q[:,0] = q[:,1]
        q[:,-1] = q[:,-2]
        
    def BC_zeros(self,q):
        '''apply current boundary conditions'''
        #zeros for walls
        q[0,:] = 0
        q[-1,:] = 0
        q[:,0] = 0
        q[:,-1] = 0

    def BC_pinning(self,q):
        q[0,self.l_intrfc-self.mp:self.l_intrfc+self.mp-1] = -q[1,self.l_intrfc+self.mp-1:self.l_intrfc-self.mp:-1]
        q[0,self.r_intrfc-self.mp:self.r_intrfc+self.mp-1] = -q[1,self.r_intrfc+self.mp-1:self.r_intrfc-self.mp:-1]

        #check BC for pinning is correct
        # print(q[0:2,self.l_intrfc-self.mp:self.l_intrfc+self.mp])
        # print(q[0:2,self.r_intrfc-self.mp:self.r_intrfc+self.mp])

    def update_all(self,cs):
        #COMMONLY USED GRADIENTS
        #grad_phi
        dphi_dx,dphi_dy = self.dev_x(self.phi),self.dev_y(self.phi)
        
        #PRELIMINARY CALCULATIONS FOR UPDATED VARIABLES
        #calculate mu_phi
        self.mu_phi[cs] = self.beta*(self.phi[cs]**3 - self.phi[cs]) - self.kappa*self.lap(self.phi)
        self.BC_standard(self.mu_phi)

        #calculate u_stars
        self.ux_star[cs] = self.ux[cs] + self.dt*(-self.ux[cs]*self.dev_x(self.ux) - self.uy[cs]*self.dev_y(self.ux) + self.eta*self.lap(self.ux) - self.phi[cs]*self.dev_x(self.mu_phi))
        self.uy_star[cs] = self.uy[cs] + self.dt*(-self.ux[cs]*self.dev_x(self.uy) - self.uy[cs]*self.dev_y(self.uy) + self.eta*self.lap(self.uy) - self.phi[cs]*self.dev_y(self.mu_phi))
        self.BC_zeros(self.ux_star)
        self.BC_zeros(self.uy_star)

        rhs = self.dx**2 * (self.rho/self.dt) * (self.dev_x(self.ux_star) + self.dev_y(self.uy_star))

        #pressure convergence loop
        for _ in range(self.nt_pcl):
            self.p[cs] = (self.p[2:,1:-1] + self.p[0:-2,1:-1] + self.p[1:-1,2:] + self.p[1:-1,0:-2] - rhs)/4
            self.BC_standard(self.p)

        #UPDATE VARIABLES
        #update phi
        self.phi[cs] += self.dt*(-self.ux[cs]*dphi_dx - self.uy[cs]*dphi_dy + self.M*self.lap(self.mu_phi)) 
        self.BC_standard(self.phi)
        self.BC_pinning(self.phi)
        #evaporation BC
        if self.evap:
            self.phi[self.edge] = -1.3

        #update velocities
        self.ux[cs] = self.ux_star[cs] - (self.dt/self.rho)*self.dev_x(self.p)
        self.uy[cs] = self.uy_star[cs] - (self.dt/self.rho)*self.dev_y(self.p)
        self.BC_zeros(self.ux)
        self.BC_zeros(self.uy)

    def run(self):
        '''running loop'''
        #indexes [1:-1,1:-1], to keep from rewriting it and taking up space
        #cs = central slice
        cs = (slice(1,-1),slice(1,-1))
        #run entire sim
        for i in range(self.Nt + 1):
            self.update_all(cs)

            if i == self.start:
                self.evap = True

            #save files
            if i % self.save == 0:
                np.savetxt(self.file_path +'/phi'+str(i)+'.txt',self.phi)
                np.savetxt(self.file_path +'/mu'+str(i)+'.txt',self.mu_phi)
                np.savetxt(self.file_path +'/ux'+str(i)+'.txt',self.ux)
                np.savetxt(self.file_path +'/uy'+str(i)+'.txt',self.uy)

#class definition for structure
class Surfactant_alternative:
    def __init__(self,
                 dt = 0.001, dx = 0.5, Nx = 64, Ny = 64, Nt = int(1e6), 
                 M = 3.0, beta = 2.0, kappa = 1.0,  eta = 1.0, 
                 name = 'new dx_0.5'):
        ''' initalise every variable/array '''
        #set up folder to save data to
        self.name = name
        self.crash = False

        try:
            os.mkdir('./'+name+' data')
        except OSError:
            print("folder name in use")
            self.crash = True

        #time
        self.dt = dt
        self.Nt = Nt
        self.save = int(Nt / 100)

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
        kx = np.fft.fftfreq(Nx,d=dx) * 2 * np.pi
        ky = np.fft.fftfreq(Ny,d=dx) * 2 * np.pi
        self.KY,self.KX = np.meshgrid(ky,kx)
        self.k2 = self.KX**2 + self.KY**2
        self.k4 = self.k2 * self.k2

        #constants
        self.M = M
        self.beta = beta
        self.kappa = kappa
        self.eta = eta

        #scalar field array definitions
        self.phi = np.zeros((Nx,Ny))
        self.mu_phi = np.zeros((Nx,Ny))

        #vector field array definitions
        self.u_x = np.zeros((Nx,Ny))
        self.u_y = np.zeros((Nx,Ny))
        self.u_x_k = np.zeros((Nx,Ny),dtype = complex)
        self.u_y_k = np.zeros((Nx,Ny),dtype = complex)
        self.f_x = np.zeros((Nx,Ny))
        self.f_y = np.zeros((Nx,Ny))

        #initalise phi
        self.init_phi_many()
        # self.init_flat()

    def init_flat(self):
        print("half")
        half = int(self.Nx/2)
        self.phi[:,:] = 1.0
        self.phi[:half,:] = -1.0

    def init_phi_many(self):
            '''Initialize phi with multiple small drops'''
            # Initialize the grid with gas (phi == -1)
            print("many")
            self.phi[:, :] = -1.0
            size = self.Nx - 2
            center = size // 2
            off_center = size // 4
            radius = 9

            # Function to place a circle on the array
            def place_circle(xc, yc, r):
                for x in range(size):
                    for y in range(size):
                        if (x - xc) ** 2 + (y - yc) ** 2 < r ** 2:
                            self.phi[x+1, y+1] = 1.0

            # Place the corner circles
            corner_positions = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
            for (xc, yc) in corner_positions:
                place_circle(xc, yc, radius)

            # Place the top and bottom semicircles
            place_circle(center, 0, radius)       # Top semicircle
            place_circle(center, size - 1, radius)  # Bottom semicircle

            # Place the left and right semicircles
            place_circle(0, center, radius,)       # Left semicircle
            place_circle(size - 1, center, radius)   # Right semicircle

            # Place circle in the centre
            place_circle(center, center, radius)

            # Place circles in the gaps between
            place_circle(off_center,off_center,radius)
            place_circle(off_center+center,off_center,radius)
            place_circle(off_center,off_center+center,radius)
            place_circle(off_center+center,off_center+center,radius)

            self.BC_standard(self.phi)

            # Uncomment the following lines if you want to visualize the result
            plt.pcolormesh(self.X, self.Y, self.phi)
            plt.show()

    def dev_x(self,q):
        return (q[2:,1:-1] - q[0:-2,1:-1]) / (2*self.dx)
    
    def dev_y(self,q):
        return (q[1:-1,2:] - q[1:-1,0:-2]) / (2*self.dx)
    
    def grad(self,q):
        '''calculates the gradient of a scalar field (nabla(q)) '''
        dq_dx = self.dev_x(q)
        dq_dy = self.dev_y(q)
        grad = [dq_dx,dq_dy]

        return grad
    
    def lap(self,q):
        return (q[2:,1:-1] + q[0:-2,1:-1] + q[1:-1,2:] + q[1:-1,0:-2]- 4*q[1:-1,1:-1]) / (self.dx*self.dx)
    
    def BC_standard(self,q):
        '''apply standard boundary conditions'''
        #standard = periodic on y walls, neumann on x walls
        q[0,:] = q[-2,:]
        q[-1,:] = q[1,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def BC_zeros(self,q):
        '''apply current boundary conditions'''
        #zeros for walls, periodic in y
        q[0,:] = q[-2,:]
        q[-1,:] = q[1,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def update_all(self,cs):
        #COMMONLY USED GRADIENTS
        #grad_phi
        dphi_dx,dphi_dy = self.grad(self.phi)
        
        #PRELIMINARY CALCULATIONS FOR UPDATED VARIABLES
        #calculate mu_phi
        self.mu_phi[cs] = self.beta*(self.phi[cs]**3 - self.phi[cs]) -self.kappa*self.lap(self.phi)
        self.BC_standard(self.mu_phi)

        #calculate f
        self.f_x[cs] = -self.phi[cs]*self.dev_x(self.mu_phi) 
        self.f_y[cs] = -self.phi[cs]*self.dev_y(self.mu_phi) 
        self.BC_zeros(self.f_x)
        self.BC_zeros(self.f_y)

        #UPDATE VARIABLES
        #update phi
        self.phi[cs] += self.dt*(-self.u_x[cs]*dphi_dx - self.u_y[cs]*dphi_dy + self.M*self.lap(self.mu_phi)) 
        self.BC_standard(self.phi)

        #create flipped f
        # fx = np.zeros((self.Nx,self.Ny))
        # fx[:self.Nx,:] = self.f_x
        # fx[self.Nx:,:] = -self.f_x[::-1,:]

        # fy = np.zeros((self.Nx,self.Ny))
        # fy[:self.Nx,:] = self.f_y
        # fy[self.Nx:,:] = -self.f_y[::-1,:]
        
        fx = self.f_x
        fy = self.f_y

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

        #run entire sim
        for i in range(self.Nt + 1):
            self.update_all(cs)

            #save files
            if i % self.save == 0:
                print(np.sum(self.phi[cs]*self.dx*self.dx))
                np.savetxt(f'./'+self.name+' data/phi'+str(i)+'.txt',self.phi)
                
#class definition for structure
class Surfactant:
    def __init__(self,
                 dt = 0.01, dx = 0.5, Nx = 512, Ny = 256, Nt = int(1e6), 
                 M = 3.0, beta = 2.0, kappa = 1.0, xl = 0.1, kBT = 1.0,
                 B = 0.5, gamma_r = 0.01, gamma_t = 0.01, eta = 1.0, 
                 Np = 500,name = 'new dx_0.5',load=False, load_no = 0, init_type = 1):
        ''' initalise every variable/array '''
        #set up folder to save data to
        self.name = name
        self.type = init_type
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
        self.save = int(Nt / 100)

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
        if self.type:
            kx = np.fft.fftfreq(2*Nx,d=dx) * 2 * np.pi
        else:
            kx = np.fft.fftfreq(Nx,d=dx) * 2 * np.pi
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
        if self.type:
            self.u_x_k = np.zeros((2*Nx,Ny),dtype = complex)
            self.u_y_k = np.zeros((2*Nx,Ny),dtype = complex)
        else:
            self.u_x_k = np.zeros((Nx,Ny),dtype = complex)
            self.u_y_k = np.zeros((Nx,Ny),dtype = complex)
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
            if self.type:
                self.init_phi_flat()
            else:
                self.init_phi_many()
            #initialise c
            self.init_c()
        else:
            self.phi = np.loadtxt(f'./'+name+' data/phi'+str(load_no)+'.txt')
            self.c = np.loadtxt(f'./'+name+' data/c'+str(load_no)+'.txt')
            self.px = np.loadtxt(f'./'+name+' data/px'+str(load_no)+'.txt')
            self.py = np.loadtxt(f'./'+name+' data/py'+str(load_no)+'.txt')

    def init_phi_many(self):
        '''Initialize phi with multiple small drops'''
        # Initialize the grid with gas (phi == -1)
        print("many")
        self.phi[:, :] = -1.0
        size = self.Nx - 2
        center = size // 2
        off_center = size // 4
        radius = 9

        # Function to place a circle on the array
        def place_circle(xc, yc, r):
            for x in range(size):
                for y in range(size):
                    if (x - xc) ** 2 + (y - yc) ** 2 < r ** 2:
                        self.phi[x+1, y+1] = 1.0

        # Place the corner circles
        corner_positions = [(0, 0), (0, size - 1), (size - 1, 0), (size - 1, size - 1)]
        for (xc, yc) in corner_positions:
            place_circle(xc, yc, radius)

        # Place the top and bottom semicircles
        place_circle(center, 0, radius)       # Top semicircle
        place_circle(center, size - 1, radius)  # Bottom semicircle

        # Place the left and right semicircles
        place_circle(0, center, radius,)       # Left semicircle
        place_circle(size - 1, center, radius)   # Right semicircle

        # Place circle in the centre
        place_circle(center, center, radius)

        # Place circles in the gaps between
        place_circle(off_center,off_center,radius)
        place_circle(off_center+center,off_center,radius)
        place_circle(off_center,off_center+center,radius)
        place_circle(off_center+center,off_center+center,radius)

        self.BC_standard(self.phi)
        # # Uncomment the following lines if you want to visualize the result
        # plt.pcolormesh(self.X, self.Y, self.phi)
        # plt.show()

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
        if self.type:
            q[0,:] = q[1,:]
            q[-1,:] = q[-2,:]
        else:
            q[0,:] = q[-2,:]
            q[-1,:] = q[1,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def BC_neg_standard(self,q):
        '''apply standard boundary conditions'''
        #negative standard = periodic on y walls, negative neumann on x walls
        if self.type:
            q[0,:] = -q[1,:]
            q[-1,:] = -q[-2,:]
        else:
            q[0,:] = q[-2,:]
            q[-1,:] = q[1,:]

        q[:,0] = q[:,-2]
        q[:,-1] = q[:,1]

    def BC_zeros(self,q):
        '''apply current boundary conditions'''
        #zeros for walls, periodic in y
        if self.type:
            q[0,:] = 0
            q[-1,:] = 0
        else:
            q[0,:] = q[-2,:]
            q[-1,:] = q[1,:]

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
        #######################################################################################################################################################
        #PRELIMINARY CALCULATIONS FOR UPDATED VARIABLES
        #######################################################################################################################################################
        #calculate mu_phi
        self.c_p_x[cs] = self.c[cs]*self.p_x[cs]
        self.c_p_y[cs] = self.c[cs]*self.p_y[cs]
        self.BC_standard(self.c_p_x)
        self.BC_standard(self.c_p_y)
        
        self.mu_phi[cs] = self.beta*(self.phi[cs]**3 - self.phi[cs]) -self.kappa*self.lap(self.phi) - self.xl*self.div(self.c_p_x,self.c_p_y)
        self.BC_standard(self.mu_phi)
        #######################################################################################################################################################
        #calculate 2/3 of mu_c
        self.mu_c[cs] = self.kBT*(self.p_x[cs]**2 + self.p_y[cs]**2) + self.xl*(self.p_x[cs]*dphi_dx + self.p_y[cs]*dphi_dy)
        self.BC_standard(self.mu_c)
        #######################################################################################################################################################
        #calculate h (without c)
        self.h_x[cs] = 2*self.kBT*self.p_x[cs] + self.xl*dphi_dx
        self.h_y[cs] = 2*self.kBT*self.p_y[cs] + self.xl*dphi_dy
        self.BC_standard(self.h_x)
        self.BC_standard(self.h_y)
        #######################################################################################################################################################
        #calculate omega
        self.omega_xy = 0.5*(du_y_dx - du_x_dy)
        self.omega_yx = 0.5*(du_x_dy - du_y_dx)
        #######################################################################################################################################################
        #calculate D
        self.D_xx = du_x_dx
        self.D_xy = 0.5*(du_y_dx + du_x_dy)
        self.D_yy = du_y_dy
        #######################################################################################################################################################
        #calculate sigma
        c_h_x = self.c*self.h_x
        c_h_y = self.c*self.h_y
        self.BC_standard(c_h_x)
        self.BC_standard(c_h_y)

        self.sigma_xx[cs] = (self.B/2) * (self.p_x[cs]*c_h_x[cs])
        self.sigma_xy[cs] = (self.p_x[cs]*c_h_y[cs])*((self.B/4) - 0.5) + (self.p_y[cs]*c_h_x[cs])*((self.B/4)+0.5)
        self.sigma_yx[cs] = (self.p_y[cs]*c_h_x[cs])*((self.B/4) - 0.5) + (self.p_x[cs]*c_h_y[cs])*((self.B/4)+0.5)
        self.sigma_yy[cs] = (self.B/2) * (self.p_y[cs]*c_h_y[cs])
        
        self.BC_standard(self.sigma_xx)
        self.BC_standard(self.sigma_xy)
        self.BC_standard(self.sigma_yx)
        self.BC_standard(self.sigma_yy)
        #######################################################################################################################################################
        #calculate f
        dh_x_dx, dh_x_dy = self.grad(c_h_x)
        dh_y_dx, dh_y_dy = self.grad(c_h_y)
        
        self.f_x[cs] = -self.phi[cs]*self.dev_x(self.mu_phi) - self.c[cs]*self.dev_x(self.mu_c) - self.kBT*dc_dx - self.p_x[cs]*dh_x_dx \
            - self.p_y[cs]*dh_y_dx + self.dev_x(self.sigma_xx) + self.dev_y(self.sigma_yx)
        self.f_y[cs] = -self.phi[cs]*self.dev_y(self.mu_phi) - self.c[cs]*self.dev_y(self.mu_c) - self.kBT*dc_dy - self.p_x[cs]*dh_x_dy \
            - self.p_y[cs]*dh_y_dy + self.dev_x(self.sigma_xy) + self.dev_y(self.sigma_yy)
        
        self.BC_zeros(self.f_x)
        self.BC_zeros(self.f_y)
        #######################################################################################################################################################
        #UPDATE VARIABLES
        #######################################################################################################################################################
        #update phi
        self.phi[cs] += self.dt*(-self.u_x[cs]*dphi_dx - self.u_y[cs]*dphi_dy + self.M*self.lap(self.mu_phi)) 
        self.BC_standard(self.phi)
        #######################################################################################################################################################
        #update c
        self.dmu_c_dx[cs],self.dmu_c_dy[cs] = (self.c[cs]/self.gamma_t)*self.grad(self.mu_c)
        self.BC_neg_standard(self.dmu_c_dx)
        self.BC_neg_standard(self.dmu_c_dy)
        self.c[cs] += self.dt*(-self.u_x[cs]*dc_dx - self.u_y[cs]*dc_dy + (self.kBT/self.gamma_t)*self.lap(self.c) + self.div(self.dmu_c_dx,self.dmu_c_dy))    
        self.BC_standard(self.c)   
        #######################################################################################################################################################
        #update px
        self.p_x[cs] += self.dt*(-self.u_x[cs]*self.dev_x(self.p_x) - self.u_y[cs]*self.dev_y(self.p_x) + self.omega_xy*self.p_y[cs] \
                                 - (self.B/2)*(self.D_xx*self.p_x[cs] + self.D_xy*self.p_y[cs]) - (1/(2*self.gamma_r))*self.h_x[cs])
        self.BC_standard(self.p_x)
        #######################################################################################################################################################
        #update py
        self.p_y[cs] += self.dt*(-self.u_x[cs]*self.dev_x(self.p_y) - self.u_y[cs]*self.dev_y(self.p_y) + self.omega_yx*self.p_x[cs] \
                                 - (self.B/2)*(self.D_xy*self.p_x[cs] + self.D_yy*self.p_y[cs]) - (1/(2*self.gamma_r))*self.h_y[cs])
        self.BC_standard(self.p_y)
        #######################################################################################################################################################
        #create flipped f (if walls  are on)
        if self.type:
            fx = np.zeros((2*self.Nx,self.Ny))
            fx[:self.Nx,:] = self.f_x
            fx[self.Nx:,:] = -self.f_x[::-1,:]

            fy = np.zeros((2*self.Nx,self.Ny))
            fy[:self.Nx,:] = self.f_y
            fy[self.Nx:,:] = -self.f_y[::-1,:]
        else:
            fx = self.f_x
            fy = self.f_y
        #######################################################################################################################################################
        #fourier parameters
        fx_k = np.fft.fft2(fx,norm='ortho')*self.dx
        fy_k = np.fft.fft2(fy,norm='ortho')*self.dx

        #preliminary calculations for velocity
        k_f = (self.KX*fx_k) + (self.KY*fy_k)
    
        #to avoid a divide by 0
        ind = (self.k2 != 0)
        #index that is dodged remains zeros anyways (i think)
        ######################################################################################################################################################
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

            #save files
            if i % self.save == 0:
                print(np.sum(self.c[cs]*self.dx*self.dx))
                print(np.sum(self.phi[cs]*self.dx*self.dx))
                np.savetxt(f'./'+self.name+' data/phi'+str(i+self.load_no)+'.txt',self.phi)
                np.savetxt(f'./'+self.name+' data/c'+str(i+self.load_no)+'.txt',self.c)
                np.savetxt(f'./'+self.name+' data/px'+str(i+self.load_no)+'.txt',self.p_x)
                np.savetxt(f'./'+self.name+' data/py'+str(i+self.load_no)+'.txt',self.p_y)


def run_simulation(simulation_class, *args, **kwargs):
    """
    Runs a simulation, handles exceptions, and frees memory afterward.
    :param simulation_class: Class of the simulation (e.g., Surfactant, Droplet).
    :param args: Positional arguments for the simulation class.
    :param kwargs: Keyword arguments for the simulation class.
    """
    try:
        sim = simulation_class(*args, **kwargs)
        if not sim.crash:  # Check if it crashed
            sim.run()  # Run the simulation
        print(f"{kwargs.get('name', 'Simulation')} passed")
    except Exception as e:
        print(f"{kwargs.get('name', 'Simulation')} crashed: {e}")
    finally:
        # Explicitly delete the object and force garbage collection
        del sim
        gc.collect()

# Running the simulations


# run_simulation(Surfactant_alternative, name='test2', dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt = int(50e6))

# run_simulation(Surfactant, name='test', init_type = 0, dx = 1.0, Nx = 64, Ny = 64, Np = 500, dt = 0.001, xl = 0.001, Nt = int(1e5))

# run_simulation(Surfactant_alternative,name='tanh test',Nt=int(1e5))

run_simulation(Surfactant,name='epsilon=0.9', dx = 0.25, Np = 400, dt = 0.0001, xl = 0.9 , Nt = int(10e5),load=True,load_no = int(19.0e5))
# run_simulation(Surfactant,name='epsilon=0.7', dx = 0.25, Np = 400, dt = 0.0001, xl = 0.7 , Nt = int(5e5),load=True,load_no = int(5.5e5))
# run_simulation(Surfactant,name='epsilon=0.5', dx = 0.25, Np = 400, dt = 0.0001, xl = 0.5 , Nt = int(5e5),load=True,load_no = int(5.5e5))
# run_simulation(Surfactant,name='epsilon=0.3', dx = 0.25, Np = 400, dt = 0.0001, xl = 0.3 , Nt = int(5e5),load=True,load_no = int(5.5e5))
# run_simulation(Surfactant,name='epsilon=0.1', dx = 0.25, Np = 400, dt = 0.0001, xl = 0.1 , Nt = int(5e5),load=True,load_no = int(5.5e5))
# run_simulation(Surfactant,name='epsilon=0.01',dx = 0.25, Np = 400, dt = 0.0001, xl = 0.01, Nt = int(5e5),load=True,load_no = int(5.5e5))

# run_simulation(Surfactant_alternative, name='ultrastable emulsion - no surfactant vrs', dx = 1.0, Nx = 64, Ny = 64, dt = 0.001,Nt=int(0.5e6))

# run_simulation(Surfactant, name='ultrastable emulsion 0.5 - 500',  xl=0.5, Np = 500,  init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 0.5 - 1000', xl=0.5, Np = 1000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 0.5 - 1500', xl=0.5, Np = 1500, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))

# run_simulation(Surfactant, name='ultrastable emulsion 1.0 - 500',  xl=1.0, Np = 500,  init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 1.0 - 1000', xl=1.0, Np = 1000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 1.0 - 1500', xl=1.0, Np = 1500, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))

# run_simulation(Surfactant, name='ultrastable emulsion 2.0 - 500',  xl=2.0, Np = 500,  init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 2.0 - 1000', xl=2.0, Np = 1000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 2.0 - 1500', xl=2.0, Np = 1500, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))

# run_simulation(Surfactant, name='ultrastable emulsion 1.5 - 500',  xl=1.5, Np = 500,  init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 1.5 - 1000', xl=1.5, Np = 1000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 1.5 - 1500', xl=1.5, Np = 1500, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))

# run_simulation(Surfactant, name='ultrastable emulsion 0.5 - 2000', xl=0.5, Np = 2000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 1.0 - 2000', xl=1.0, Np = 2000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 1.5 - 2000', xl=1.5, Np = 2000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))
# run_simulation(Surfactant, name='ultrastable emulsion 2.0 - 2000', xl=2.0, Np = 2000, init_type=0, dx = 1.0, Nx = 64, Ny = 64, dt = 0.001, Nt=int(0.5e6))

# run_simulation(Droplet, name='droplet evaporation')
# run_simulation(Surfactant, name='x range test', xl=0.1, init_type=1, Nx=513, Ny=257)

