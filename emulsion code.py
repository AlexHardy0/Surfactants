import numpy as np
import matplotlib.pyplot as plt
import os
import gc
import traceback

class ShearFlow:
    def __init__(self,
                 dx=1.0,dt=0.001,Nx=64,Ny=64,Nt=int(1e6),Nt_save=int(1e5),
                 M=1.0,beta=1.0,kappa=1.0,eta=1.0,u_mag = 0.0,xl=0.1,F=0.01,
                 kBT=1.0,B=0.5,gamma_r=0.1,gamma_t=0.1,Np=1000,no_drops = 0,
                 name='test', load=False,load_no=0):
        
        #prepare for saving
        self.name = name
        self.crash = False
        self.load_no = load_no
        
        try:
            #make saving folder
            os.mkdir(f'./{name}')
        except OSError:
            #if folder exists
            if not load:
                #and not loading from it, stop the program
                self.crash = True
                # pass
        
        #region store all inputted values
        
        #system parameters
        self.dx = dx
        self.dt = dt
        self.Nx = Nx
        self.Ny = Ny
        self.Np = Np
        self.Nt = Nt
        self.Nt_save = Nt_save
    
        #physics parameters
        self.M = M
        self.beta = beta
        self.kappa = kappa
        self.eta = eta
        self.xl = xl
        self.kBT = kBT
        self.B = B
        self.gamma_r = gamma_r
        self.gamma_t = gamma_t

        #endregion

        #meshgrid for plotting
        x = np.arange(0.0,dx*Nx,dx)
        y = np.arange(0.0,dx*Ny,dx)
        self.Y,self.X = np.meshgrid(y,x)

        #region set up all arrays

        #fourier space
        kx = np.fft.fftfreq(Nx,d=dx)*2*np.pi
        ky = np.fft.fftfreq(2*Ny,d=dx)*2*np.pi
        kx[0] = 0.01
        ky[0] = 0.01
        KY,KX = np.meshgrid(ky,kx)
        self.K = np.stack((KX,KY))
        self.k2 = KX**2 + KY**2
        self.k4 = self.k2 * self.k2

        #scalar fields
        self.phi = np.zeros((Nx+2,Ny+2))
        self.mu_phi = np.zeros((Nx+2,Ny+2))
        self.c = np.zeros((Nx+2,Ny+2))
        self.mu_c = np.zeros((Nx+2,Ny+2))

        #vector fields
        self.u = np.zeros((2,Nx+2,Ny+2))
        self.u_k = np.zeros((2,Nx,2*Ny),dtype=complex)
        self.f = np.zeros((2,Nx,Ny))
        self.p = np.zeros((2,Nx+2,Ny+2))
        self.h = np.zeros((2,Nx+2,Ny+2))
        self.J = np.zeros((2,Nx+2,Ny+2))

        #derivatives of fields
        self.dphi = np.zeros((2,Nx,Ny))
        self.dc = np.zeros((2,Nx,Ny))
        self.du_x = np.zeros((2,Nx,Ny))
        self.du_y = np.zeros((2,Nx,Ny))
        self.dp_x = np.zeros((2,Nx,Ny))
        self.dp_y = np.zeros((2,Nx,Ny))
        self.dmu_c = np.zeros((2,Nx,Ny))
        self.dmu_phi = np.zeros((2,Nx,Ny))
        self.dh_dxs = np.zeros((2,Nx,Ny))
        self.dh_dys = np.zeros((2,Nx,Ny))

        #combinations of fields
        self.c_p = np.zeros((2,Nx+2,Ny+2))
        self.p_dh = np.zeros((2,Nx,Ny))
        self.dsigma = np.zeros((2,Nx,Ny))
        self.u_dp = np.zeros((2,Nx,Ny))
        self.p_D = np.zeros((2,Nx,Ny))
        self.u_du = np.zeros((2,Nx,Ny))

        #matrix definitions
        self.omega = np.zeros((2,Nx,Ny))
        self.D_x = np.zeros((2,Nx,Ny))
        self.D_y = np.zeros((2,Nx,Ny))
        
        self.sigma_xx = np.zeros((Nx+2,Ny+2))
        self.sigma_xy = np.zeros((Nx+2,Ny+2))
        self.sigma_yx = np.zeros((Nx+2,Ny+2))
        self.sigma_yy = np.zeros((Nx+2,Ny+2))

        #endregion

        #initalise the system
        if not load:
            #initalise phi
            self.init_phi(no_drops)
            #initalise c
            c_init = self.Np/(Nx*Ny*dx*dx)
            self.c[1:-1,1:-1] = c_init
            self.BC_standard(self.c)

            # plt.pcolormesh(self.X,self.Y,self.c)
            # plt.colorbar()
            # plt.show()
        else:
            #load from folder
            data = np.load(f'./{name}/step_{load_no}.npz')
            self.phi = data['phi']
            self.c = data['c']
            self.p[0] = data['px']
            self.p[1] = data['py']
            self.u[0] = data['ux']
            self.u[1] = data['uy']


    def init_phi(self,no_drops):
        # Initialize the grid with gas
        self.phi[:, :] = -1.0

        xsize = self.Nx
        ysize = self.Ny

        # Function to place a circle in phi
        def place_circle(xc, yc, r):
            '''
            function for placing small drops in phi

            Parameters:
            xc : center x coord
            yc : center y coord
            r  : radius of drop
            '''

            for x in range(xsize):
                for y in range(ysize):
                    if (x - xc) ** 2 + (y - yc) ** 2 < r ** 2:
                        self.phi[x+1, y+1] = 1.0

        # Place drop in the centre     
        if no_drops == 0:
            half = (self.Ny+4) // 2
            self.phi[:,half:] = 1.0
        elif no_drops == 1:   
            # small drop parameters
            xcenter = xsize // 2
            ycenter = ysize // 2
            radius = ysize // 3
            place_circle(xcenter, ycenter, radius)
        elif no_drops ==2:
            radius = ysize // 6
            xcenter = xsize // 4
            ycenter = ysize // 2
            yoffset = ysize // 5
            place_circle(xcenter,ycenter-yoffset,radius)
            place_circle(xcenter,ycenter+yoffset,radius)
        else:
            size = self.Nx
            center = size // 2
            off_center = size // 4
            radius = size // 7
            # place the top and bottom drops
            place_circle(center, 0, radius)       
            place_circle(center, size - 1, radius) 

            # place the left and right drops
            place_circle(0, center, radius,)      
            place_circle(size -1, center, radius)   

            # place drop in the centre
            place_circle(center, center, radius)

            # place drops in the gaps between
            place_circle(off_center,off_center,radius)
            place_circle(off_center+center,off_center,radius)
            place_circle(off_center,off_center+center,radius)
            place_circle(off_center+center,off_center+center,radius)
            place_circle(size-1,size-1,radius)
            place_circle(size-1,0,radius)
            place_circle(0,size-1,radius)
            place_circle(0,0,radius)

        # apply boundary conditions
        self.BC_standard(self.phi)

        # plt.pcolormesh(self.X,self.Y,self.phi[1:-1,1:-1])
        # plt.show()

    def dev_x(self,q):
        # dev = (q[2:,1:-1] - q[0:-2,1:-1]) / (2*self.dx)
        dev = ((    q[2:,2:]   -     q[0:-2,2:]) \
            + (4.0*q[2:,1:-1] - 4.0*q[0:-2,1:-1]) \
			+ (    q[2:,0:-2] -     q[0:-2,0:-2]))/(12*self.dx)
        return dev

    def dev_ux(self,q):
        '''derivative in x direction, q: array to differentiate'''
        right = np.where(self.u[1] > 0.00001,1,0)[1:-1,1:-1]
        left = np.where(self.u[1] < -0.00001,1,0)[1:-1,1:-1]
        centre = np.where((self.u[1] > -0.00001) & (self.u[1] < 0.00001),1,0)[1:-1,1:-1]
        dev = right*((q[2:,1:-1]-q[1:-1,1:-1])/self.dx) + left*((q[1:-1,1:-1]-q[0:-2,1:-1])/self.dx)+  centre*((q[2:,1:-1] - q[0:-2,1:-1]) / (2*self.dx))
        
        return dev
    
    def dev_y(self,q):
        # dev = (q[1:-1,2:] - q[1:-1,0:-2]) / (2*self.dx)
        dev = ((q[2:,2:]   -     q[2:,0:-2]) \
			+ (4.0*q[1:-1,2:] - 4.0*q[1:-1,0:-2]) \
			+ (q[0:-2,2:] -     q[0:-2,0:-2]))/(12*self.dx)
        return dev

    def dev_uy(self,q):
        '''derivative in y direction, q: array to differentiate'''
        up = np.where(self.u[1] > 0.00001,1,0)[1:-1,1:-1]
        down = np.where(self.u[1] < -0.00001,1,0)[1:-1,1:-1]
        centre = np.where((self.u[1] > -0.00001) & (self.u[1] < 0.00001),1,0)[1:-1,1:-1]
        dev = up*((q[1:-1,2:]-q[1:-1,1:-1])/self.dx) + down*((q[1:-1,1:-1]-q[1:-1,0:-2])/self.dx) + centre*((q[1:-1,2:] - q[1:-1,0:-2]) / (2*self.dx))
        
        return dev
    
    def lap(self,q):
        '''calculates grad^2(q)'''
        # lap = (q[2:,1:-1] + q[0:-2,1:-1] + q[1:-1,2:] + q[1:-1,0:-2]- 4*q[1:-1,1:-1]) / (self.dx*self.dx)
        lap = ((q[0:-2,2:]   +  4.0*q[1:-1,2:]   +     q[2:,2:]) \
			+ (4.0*q[0:-2,1:-1] - 20.0*q[1:-1,1:-1] + 4.0*q[2:,1:-1]) \
			+ (q[0:-2,0:-2] +  4.0*q[1:-1,0:-2] +     q[2:,0:-2]))/(6*self.dx*self.dx)
        return lap
    
    def div(self,qx,qy):
        '''calculates divergence of a vector field (nabla . q) '''
        # div = (qx[2:,1:-1] - qx[0:-2,1:-1] + qy[1:-1,2:] - qy[1:-1,0:-2]) / (2*self.dx)
        div = self.dev_x(qx) + self.dev_y(qy)

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
        q[0,:] = q[-2,:]
        q[-1,:] = q[1,:]

        q[:,0] = q[:,1]
        q[:,-1] = q[:,-2]

    def BC_neg_standard(self,q):
        '''apply negative standard boundary conditions to array q'''
        q[0,:] = q[-2,:]
        q[-1,:] = q[1,:]

        q[:,0] = -q[:,1]
        q[:,-1] =-q[:,-2]

    def BC_zeros(self,q):
        '''apply zero boundary conditions to array q'''
        q[0,:] = q[-2,:]
        q[-1,:] = q[1,:]

        q[:,0] = 0
        q[:,-1] = 0

    def run(self):
        #indexes for slicing
        cs = (slice(1,-1),slice(1,-1))
        css = (slice(None),slice(1,-1),slice(1,-1))
        rcss = (slice(-1,None,-1),slice(1,-1),slice(1,-1))

        for i in range(10000):
            self.mu_phi[cs] = self.beta*(self.phi[cs]*self.phi[cs]*self.phi[cs] - self.phi[cs]) - self.kappa*self.lap(self.phi)- self.xl*self.div(self.c_p[0],self.c_p[1])
            self.BC_standard(self.mu_phi)

            self.phi[cs] += self.dt*(self.M*self.lap(self.mu_phi)) 
            self.BC_standard(self.phi)

            # calculate 2/3 of mu_c
            self.mu_c[cs] = self.kBT*(self.dot(self.p[css],self.p[css])) + self.xl*(self.dot(self.p[css],self.dphi))
            self.BC_standard(self.mu_c)

            self.dmu_c[0],self.dmu_c[1] = self.dev_x(self.mu_c),self.dev_y(self.mu_c)

            # update c
            self.J[css] = (self.c[cs]/self.gamma_t)*self.dmu_c
            self.BC_neg_standard(self.J[0])
            self.BC_neg_standard(self.J[1])
            
            self.c[cs] += self.dt*((self.kBT/self.gamma_t)*self.lap(self.c) + self.div(self.J[0],self.J[1]))    
            self.BC_standard(self.c)  

        #run entire sim
        for i in range(self.Nt + 1):
            #region RUNNING LOOP
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
             #calculate mu_phi
            self.c_p = self.c*self.p
            self.BC_standard(self.c_p[0])
            self.BC_standard(self.c_p[1])

            self.mu_phi[cs] = self.beta*(self.phi[cs]*self.phi[cs]*self.phi[cs] - self.phi[cs]) - self.kappa*self.lap(self.phi)\
                - self.xl*self.div(self.c_p[0],self.c_p[1])
            self.BC_standard(self.mu_phi)
            #######################################################################################################################################################
            # calculate 2/3 of mu_c
            self.mu_c[cs] = self.kBT*(self.dot(self.p[css],self.p[css])) + self.xl*(self.dot(self.p[css],self.dphi))
            self.BC_standard(self.mu_c)

            self.dmu_c[0],self.dmu_c[1] = self.dev_x(self.mu_c),self.dev_y(self.mu_c)
            #######################################################################################################################################################
            # calculate h (without c)
            self.h[css] = 2*self.kBT*self.p[css] + self.xl*self.dphi
            self.BC_standard(self.h[0])
            self.BC_standard(self.h[1])
            #######################################################################################################################################################
            # calculate omega
            self.omega[0] = 0.5*(self.du_y[0] - self.du_x[1])
            self.omega[1] = 0.5*(self.du_x[1] - self.du_y[0])
            #######################################################################################################################################################
            # calculate D
            self.D_x[0] = self.du_x[0]
            self.D_x[1] = 0.5*(self.du_y[0] + self.du_x[1])
            self.D_y[0] = 0.5*(self.du_y[0] + self.du_x[1])
            self.D_y[1] = self.du_y[1]
            #######################################################################################################################################################
            c_h = self.c * self.h
            self.BC_standard(c_h[0])
            self.BC_standard(c_h[1])

            # calculate sigma
            p_ch = self.p*c_h
        
            self.sigma_xx[cs],self.sigma_yy[cs] = (0.5*self.B) * p_ch[css]
            self.sigma_xy[cs] = ((0.25*self.B)*self.dot(self.p[css],c_h[rcss])) - 0.5*self.neg_dot(self.p[css],c_h[rcss])
            self.sigma_yx[cs] = ((0.25*self.B)*self.dot(self.p[rcss],c_h[css])) - 0.5*self.neg_dot(self.p[rcss],c_h[css])
            
            self.BC_standard(self.sigma_xx)
            self.BC_standard(self.sigma_xy)
            self.BC_standard(self.sigma_yx)
            self.BC_standard(self.sigma_yy)
            #######################################################################################################################################################
            # dchx dx  dchy dx
            self.dh_dxs[0] = self.dev_x(c_h[0]) 
            self.dh_dxs[1] = self.dev_x(c_h[1])
            # dchx dy dchy dy 
            self.dh_dys[0] = self.dev_y(c_h[0])
            self.dh_dys[1] = self.dev_y(c_h[1])

            # calculate f
            self.dmu_phi[0] = self.dev_x(self.mu_phi)
            self.dmu_phi[1] = self.dev_y(self.mu_phi)
            self.p_dh[0] = self.dot(self.p[css],self.dh_dxs)
            self.p_dh[1] = self.dot(self.p[css],self.dh_dys)
            self.dsigma[0] = self.div(self.sigma_xx,self.sigma_xy)
            self.dsigma[1] = self.div(self.sigma_yx,self.sigma_yy)

            self.f = -self.phi[cs]*self.dmu_phi - self.c[cs]*self.dmu_c - self.kBT*self.dc - self.p_dh + self.dsigma
            self.BC_zeros(self.f[0])
            self.BC_zeros(self.f[1])

            #######################################################################################################################################################
            # UPDATE ALL FIELDS + APPLY APPROPIATE BC'S
            #######################################################################################################################################################
            # update phi
            dphi = [self.dev_ux(self.phi),self.dev_uy(self.phi)]
            self.phi[cs] += self.dt*(-self.dot(self.u[css],dphi)+ self.M*self.lap(self.mu_phi)) 
            self.BC_standard(self.phi)
            #######################################################################################################################################################
            # update c
            dc = [self.dev_x(self.c),self.dev_uy(self.c)]

            self.J[css] = (self.c[cs]/self.gamma_t)*self.dmu_c
            self.BC_neg_standard(self.J[0])
            self.BC_neg_standard(self.J[1])
            
            self.c[cs] += self.dt*(-self.dot(self.u[css],dc) + (self.kBT/self.gamma_t)*self.lap(self.c) + self.div(self.J[0],self.J[1]))    
            self.BC_standard(self.c)  
            #######################################################################################################################################################
            self.dp_x[0] = self.dev_ux(self.p[0])
            self.dp_x[1] = self.dev_uy(self.p[0])
            self.dp_y[0] = self.dev_ux(self.p[1])
            self.dp_y[1] = self.dev_uy(self.p[1])
            self.u_dp[0] = self.dot(self.u[css],self.dp_x)
            self.u_dp[1] = self.dot(self.u[css],self.dp_y)
            self.p_D[0] = self.dot(self.p[css],self.D_x)
            self.p_D[1] = self.dot(self.p[css],self.D_y)

            self.p[css] += self.dt*(-self.u_dp - self.omega*self.p[rcss] + (0.5*self.B)*self.p_D - 0.5*(1/self.gamma_r)*self.h[css])
            self.BC_standard(self.p[0])
            self.BC_standard(self.p[1])
            #######################################################################################################################################################
            #create flipped f (if walls  are on)
            f = np.stack((np.zeros((self.Nx,2*self.Ny)),np.zeros((self.Nx,2*self.Ny))))
            f[0,:,:self.Ny] = self.f[0]
            f[0,:,self.Ny:] = -self.f[0,:,::-1]

            f[1,:,:self.Ny] = self.f[1]
            f[1,:,self.Ny:] = -self.f[1,:,::-1]
            
            #######################################################################################################################################################
            # fourier parameters
            f_k = np.fft.fft2(f,norm='ortho')*self.dx

            # preliminary calculations for velocity
            k_f = f_k[0]*self.K[0] + f_k[1]*self.K[1]
        
            ######################################################################################################################################################
            # velocity calculations
            self.u_k = ((f_k/self.k2) - (self.K*k_f/self.k4))/self.eta

            # transform u back into real space
            self.u[css] = (np.fft.ifft2(self.u_k,norm='ortho').real/self.dx)[:,:,:self.Ny]

            self.BC_standard(self.u[0])
            self.BC_standard(self.u[1])

            #endregion

            #save files, continues to use the same folder when loading
            if i % self.Nt_save == 0:
                # uncomment for conservation check
                # print(np.sum(self.c[cs]*self.dx*self.dx))  #conservation check
                # print(np.sum(self.phi[cs]*self.dx*self.dx))  #conservation check
                np.savez(f'./{self.name}/step_{i+self.load_no}.npz', phi=self.phi, c=self.c, px=self.p[0], py=self.p[1], ux = self.u[0], uy =self.u[1], fx = self.f[0],fy=self.f[1])


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


# run_simulation(ShearFlow,name='emulsion no surf',  Nt=int(5e6),Nt_save=int(1e3),dt=0.0001,dx=0.5,B=0.0,xl=0.0,Np=0,   no_drops=3,Nx=128,Ny=128)
# run_simulation(ShearFlow,name='emulsion ys surf',  Nt=int(5e6),Nt_save=int(1e3),dt=0.0001,dx=0.5,B=1.0,xl=1.5,Np=1000,no_drops=3,Nx=128,Ny=128)

