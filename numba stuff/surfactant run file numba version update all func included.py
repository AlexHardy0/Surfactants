# imports
import numpy as np
import os
import gc
import cProfile
import pstats
import traceback

profiler = cProfile.Profile()
profiler.enable()

def make_folder(name,load):        
    try:
        os.mkdir('./'+name+'_data')
    except OSError:
        print("folder name in use")
        if load == False:
            # stop the simulation to avoid overwriting data, except if you are loading from that folder
            # self.crash = True
            pass

@njit
def dev_x(q,dx):
    '''derivative in x direction, q: array to differentiate'''
    dev = (q[2:,1:-1] - q[0:-2,1:-1]) / (2*dx)
    return dev
 
@njit
def dev_y(q,dx):
    '''derivative in y direction, q: array to differentiate'''
    dev = (q[1:-1,2:] - q[1:-1,0:-2]) / (2*dx)
    return dev

@njit
def lap(q,dx):
    '''calculates grad^2(q)'''
    lap = (q[2:,1:-1] + q[0:-2,1:-1] + q[1:-1,2:] + q[1:-1,0:-2]- 4*q[1:-1,1:-1]) / (dx*dx)
    return lap

@njit
def div(qx,qy,dx):
    '''calculates divergence of a vector field (nabla . q) '''
    div = (qx[2:,1:-1] - qx[0:-2,1:-1] + qy[1:-1,2:] - qy[1:-1,0:-2]) / (2*dx)

    return div

@njit
def dot(q1,q2):
    '''calculates dot product from STACKS (q1 and q2)'''
    dot = q1[0]*q2[0] + q1[1]*q2[1]
    return dot

@njit
def neg_dot(q1,q2):
    '''calculates dot product from STACKS (q1 and q2) except instead of adding it subtracts'''
    dot = q1[0]*q2[0] - q1[1]*q2[1]
    return dot

@njit
def BC_standard(q,sim_type):
    '''apply standard boundary conditions to array q'''
    #standard = periodic on y walls, neumann on x walls for flat interface
    #standard = periodic on all walls for emulsion
    if sim_type:
        q[0,:] = q[1,:]
        q[-1,:] = q[-2,:]
    else:
        q[0,:] = q[-2,:]
        q[-1,:] = q[1,:]

    q[:,0] = q[:,-2]
    q[:,-1] = q[:,1]

@njit 
def BC_neg_standard(q,sim_type):
    '''apply negative standard boundary conditions to array q'''
    #standard = periodic on y walls, negative neumann on x walls for flat interface
    #standard = periodic on all walls for emulsion
    if sim_type:
        q[0,:] = -q[1,:]
        q[-1,:] = -q[-2,:]
    else:
        q[0,:] = q[-2,:]
        q[-1,:] = q[1,:]

    q[:,0] = q[:,-2]
    q[:,-1] = q[:,1]

@njit
def BC_zeros(q,sim_type):
    '''apply zero boundary conditions to array q'''
    #standard = periodic on y walls, zero on x walls for flat interface
    #standard = periodic on all walls for emulsion
    if sim_type:
        q[0,:] = 0
        q[-1,:] = 0
    else:
        q[0,:] = q[-2,:]
        q[-1,:] = q[1,:]

    q[:,0] = q[:,-2]
    q[:,-1] = q[:,1]
 
def init_phi_many(phi,Nx,sim_type):
    '''
    Initialize phi with multiple small drops
    '''
    # Initialize the grid with gas
    print("many")
    phi[:, :] = -1.0

    # small drop parameters
    size = Nx - 2
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
                    phi[x+1, y+1] = 1.0

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
    BC_standard(phi,sim_type)
    
    # Uncomment the following lines if you want to visualize the result
    # plt.pcolormesh(self.X, self.Y, self.phi)
    # plt.show()
 
def init_phi_flat(phi,Nx):
    '''
    Initalizes the phi array as two halves of fluid, with a flat vertical interface halfway 
    '''
    print("half")
    half = int(Nx/2)
    phi[:,:] = 1.0
    phi[:half,:] = -1.0
    
    # Uncomment the following lines if you want to visualize the result
    
    #plt.pcolormesh(self.X,self.Y,self.phi)
    #plt.show()
 
def init_c(c,Np,Lx,Ly,dx,sim_type):
    '''
    Initalise c as uniform everywhere
    '''
    Lx = Lx - 2*dx
    Ly = Ly - 2*dx
    c_init = Np/(Lx*Ly)
    c[1:-1,1:-1] = c_init

    #aplpy boundary conditions
    BC_standard(c,sim_type)

    # Uncomment the following lines if you want to visualize the result
    #plt.pcolormesh(self.X,self.Y,self.c)
    #plt.show()

@jit
def update_u(uu,u_k,k2,k4,K,f,Nx,Ny,dx,eta,sim_type):
    #update velocity
    #create flipped f (if walls  are on)    
    if sim_type:
        ff = np.stack((np.zeros((2*Nx,Ny),dtype='float32'),np.zeros((2*Nx,Ny),dtype='float32')))
        ff[0,:Nx,:] = f[0]
        ff[0,Nx:,:] = -f[0,::-1,:]

        ff[1,:Nx,:] = f[1]
        ff[1,Nx:,:] = -f[1,::-1,:]
    else:
        ff = f
    #######################################################################################################################################################
    # fourier parameters
    f_k = (np.fft.fft2(ff,norm='ortho')*dx)

    # preliminary calculations for velocity
    k_f = f_k[0]*K[0] + f_k[1]*K[1]

    # index that is dodged remains zeros anyways (i think)
    ######################################################################################################################################################
    # velocity calculations
    u_k[:,:,:] = ((f_k/k2) - (K*k_f/k4))/eta

    # transform u back into real space
    uu[:,:,:] = (np.fft.ifft2(u_k,norm='ortho').real/dx)[:,:Nx,:]

def main(dt = 0.01, dx = 0.5, Nx = 512, Ny = 256, Nt = int(1e6), 
         M = 3.0, beta = 2.0, kappa = 1.0, xl = 0.1, kBT = 1.0, rho = 1.0,
         B = 0.5, gamma_r = 0.01, gamma_t = 0.01, eta = 1.0, Np = 500,
         name = 'new dx_0.5',load=False, load_no = 0, sim_type = 1):
    
    #region INIT
    ''' Initalise every variable/array to be used in the simulation as a property
        
        Parameters: all variables which can be set in the simulation, automatically set to their default value if none are specified
        they are split as:
        line 1: resolution parameters, time step, spatial res., size of x-direction, size of y-direction, Total No. of time steps
        line 2: equation parameters: M, beta, kappa, xl, kBT
        line 3: continued equation parameters: B, gamma_r, gamma_t, eta and Np, the number of particles to use
        line 4: name of the simulation folder to save to, loading information, and type of simulation run
    '''
    
    # system parameters calculated from inputs
    save = int(Nt / 100)
    Lx = int(Nx*dx)
    Ly = int(Ny*dx)

    # set up fourier space parameters, based on if walls are present or not
    if type:
        kx = np.fft.fftfreq(2*Nx,d=dx) * 2 * np.pi
    else:
        kx = np.fft.fftfreq(Nx,d=dx) * 2 * np.pi
    ky = np.fft.fftfreq(Ny,d=dx) * 2 * np.pi
    kx[0] = 0.001
    ky[0] = 0.001
    KY,KX = np.meshgrid(ky,kx)
    K = np.stack((KX,KY))
    k2 = K[0]*K[0] + K[1]*K[1]
    k4 = k2 * k2

    # scalar field array definitions
    phi = np.zeros((Nx,Ny),dtype='float32')
    mu_phi = np.zeros((Nx,Ny),dtype='float32')
    c = np.zeros((Nx,Ny),dtype='float32')
    mu_c = np.zeros((Nx,Ny),dtype='float32')
    prs = np.zeros((Nx,Ny),dtype='float32')

    # vector field array definitions
    uu = np.zeros((2,Nx,Ny),dtype='float32')
    u_star = np.zeros((2,Nx,Ny),dtype='float32')
    if type:
        u_k = np.stack((np.zeros((2*Nx,Ny),dtype = complex),np.zeros((2*Nx,Ny),dtype = complex)))
    else:
        u_k = np.stack((np.zeros((Nx,Ny),dtype = complex),np.zeros((Nx,Ny),dtype = complex)))
    p = np.zeros((2,Nx,Ny),dtype='float32')
    f = np.zeros((2,Nx,Ny),dtype='float32')
    h = np.zeros((2,Nx,Ny),dtype='float32')
    J_c = np.zeros((2,Nx,Ny),dtype='float32')

    #derivatives field array definitions
    dphi = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    dc = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    du_x = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    du_y = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    lap_u = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    dp_x = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    dp_y = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    dmu_c = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    dmu_phi = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    dh_dxs = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    dh_dys = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    dev_prs = np.zeros((2,Nx-2,Ny-2),dtype='float32')

    #combinations field array definitions
    c_p = np.zeros((2,Nx,Ny),dtype='float32')
    c_h = np.zeros((2,Nx,Ny),dtype='float32')
    p_dh = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    dsigma = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    u_dp = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    p_D = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    u_du = np.zeros((2,Nx-2,Ny-2),dtype='float32')

    # matrix array defintions
    # omega_xx amd omega_yy = 0
    #omega xy, omega yx
    omega = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    #D_xx D_xy
    D_x = np.zeros((2,Nx-2,Ny-2),dtype='float32')
    #D_xy D_yy
    D_y = np.zeros((2,Nx-2,Ny-2),dtype='float32')

    sigma_xx = np.zeros((Nx,Ny),dtype='float32')
    sigma_xy = np.zeros((Nx,Ny),dtype='float32')
    sigma_yx = np.zeros((Nx,Ny),dtype='float32')
    sigma_yy = np.zeros((Nx,Ny),dtype='float32')

    # intialize phi and c, based on simulation type + if you are loading 
    if load == False:
        # initalise phi
        if sim_type:
            init_phi_flat(phi,Nx)
        else:
            init_phi_many(phi,Nx,sim_type)
        #initialise c
        init_c(c,Np,Lx,Ly,dx,sim_type)
        uu[:,:,:] = 0 
    else:
        pass
        # #load from folder
        # phi = np.loadtxt(f'./'+name+' data/phi'+str(load_no)+'.txt')
        # c = np.loadtxt(f'./'+name+' data/c'+str(load_no)+'.txt')
        # px = np.loadtxt(f'./'+name+' data/px'+str(load_no)+'.txt')
        # py = np.loadtxt(f'./'+name+' data/py'+str(load_no)+'.txt')
#endregion

    #indexes [1:-1,1:-1], to keep from rewriting it and taking up space
    #cs = central slice
    cs = (slice(1,-1),slice(1,-1))
    #css = central slice stack
    css = (slice(None),slice(1,-1),slice(1,-1))
    #reverse css
    rcss = (slice(-1,None,-1),slice(1,-1),slice(1,-1))

    @njit
    def update_all(phi, c, p, uu, mu_phi, mu_c, h, omega, sigma_xx, sigma_xy,sigma_yx,sigma_yy, D_x,D_y, prs,
                  u_star, f, J_c, dphi ,dc, du_x, du_y, dmu_c, dmu_phi, dh_dxs, dh_dys, dp_x, dp_y, lap_u, dev_prs,
                     c_p, c_h, p_D, p_dh, dsigma, u_dp, u_du, dx, dt, M, beta, kappa, eta, rho, xl, kBT, B, gamma_r, gamma_t, sim_type):
    
        # CALCULATE COMMONLY USED GRADIENTS
        # grad_phi
        dphi[0] = dev_x(phi,dx)
        dphi[1] = dev_y(phi,dx)
        # grad_c
        dc[0] = dev_x(c,dx)
        dc[1] = dev_y(c,dx)
        # grad u_x
        du_x[0] = dev_x(uu[0],dx)
        du_x[1] = dev_y(uu[0],dx)
        # grad u_y
        du_y[0] = dev_x(uu[1],dx)
        du_y[1] = dev_y(uu[1],dx)
        
        #######################################################################################################################################################
        #PRELIMINARY CALCULATIONS FOR UPDATED VARIABLES + APPLY APPROPIATE BC'S
        #######################################################################################################################################################
        c_p[css] = c[cs]*p[css]
        BC_standard(c_p[0],sim_type)
        BC_standard(c_p[1],sim_type)
         #calculate mu_phi
        mu_phi[cs] = beta*(phi[cs]*phi[cs]*phi[cs] - phi[cs]) - kappa*lap(phi,dx)\
                - xl*div(c_p[0],c_p[1],dx)
        BC_standard(mu_phi,sim_type)
        #######################################################################################################################################################
        # calculate 2/3 of mu_c
        mu_c[cs] = kBT*(dot(p[css],p[css])) + xl*(dot(p[css],dphi))
        BC_standard(mu_c,sim_type)
        dmu_c[0],dmu_c[1] = dev_x(mu_c,dx),dev_y(mu_c,dx)
        #######################################################################################################################################################
        # calculate h (without c)
        h[css] = 2*kBT*p[css] + xl*dphi
        BC_standard(h[0],sim_type)
        BC_standard(h[1],sim_type)
        #######################################################################################################################################################
        # calculate omega
        omega[0] = 0.5*(du_y[0] - du_x[1])
        omega[1] = 0.5*(du_x[1] - du_y[0])
        
        # calculate D
        D_x[0] = du_x[0]
        D_x[1] = 0.5*(du_y[0] + du_x[1])
        D_y[0] = 0.5*(du_y[0] + du_x[1])
        D_y[1] = du_y[1]
        #######################################################################################################################################################
        c_h[css] = c[cs] * h[css]
        BC_standard(c_h[0],sim_type)
        BC_standard(c_h[1],sim_type)
        
        # calculate sigma
        p_ch = p*c_h

        sigma_xx[cs],sigma_yy[cs] = (0.5*B) * p_ch[css]
        sigma_xy[cs] = ((0.25*B)*dot(p[css],c_h[rcss])) - 0.5*neg_dot(p[css],c_h[rcss])
        sigma_yx[cs] = ((0.25*B)*dot(p[rcss],c_h[css])) - 0.5*neg_dot(p[rcss],c_h[css])

        BC_standard(sigma_xx,sim_type)
        BC_standard(sigma_xy,sim_type)
        BC_standard(sigma_yx,sim_type)
        BC_standard(sigma_yy,sim_type)
        #######################################################################################################################################################
        # dchx dx  dchy dx
        dh_dxs[0] = dev_x(c_h[0],dx) 
        dh_dxs[1] = dev_x(c_h[1],dx)
        # dchx dy dchy dy 
        dh_dys[0] = dev_y(c_h[0],dx)
        dh_dys[1] = dev_y(c_h[1],dx)

        # calculate f
        dmu_phi[0] = dev_x(mu_phi,dx)
        dmu_phi[1] = dev_y(mu_phi,dx)
        p_dh[0] = dot(p[css],dh_dxs)
        p_dh[1] = dot(p[css],dh_dys)
        dsigma[0] = div(sigma_xx,sigma_yx,dx)
        dsigma[1] = div(sigma_xy,sigma_yy,dx)

        f[css] = -phi[cs]*dmu_phi - c[cs]*dmu_c - kBT*dc - p_dh + dsigma
        BC_zeros(f[0],sim_type)
        BC_zeros(f[1],sim_type)

        #######################################################################################################################################################
        # UPDATE ALL FIELDS + APPLY APPROPIATE BC'S
        #######################################################################################################################################################
        # update phi
        phi[cs] += dt*(-dot(uu[css],dphi)+ M*lap(mu_phi,dx)) 
        BC_standard(phi,sim_type)
        #######################################################################################################################################################
        # update c
        J_c[css] = (c[cs]/gamma_t)*dmu_c
        BC_neg_standard(J_c[0],sim_type)
        BC_neg_standard(J_c[1],sim_type)
        
        c[cs] += dt*(-dot(uu[css],dc) + (kBT/gamma_t)*lap(c,dx) + div(J_c[0],J_c[1],dx)) 
        BC_standard(c,sim_type)  
        #######################################################################################################################################################
        # update p
        dp_x[0] = dev_x(p[0],dx)
        dp_x[1] = dev_y(p[0],dx)
        dp_y[0] = dev_x(p[1],dx)
        dp_y[1] = dev_y(p[1],dx)
        u_dp[0] = dot(uu[css],dp_x)
        u_dp[1] = dot(uu[css],dp_y)
        p_D[0] = dot(p[css],D_x)
        p_D[1] = dot(p[css],D_y)

        p[css] += dt*(-u_dp + omega*p[rcss] - (0.5*B)*p_D - 0.5*(1/gamma_r)*h[css])
        BC_standard(p[0],sim_type)
        BC_standard(p[1],sim_type)
        #######################################################################################################################################################
        # update_u(uu,u_k,k2,k4,K,f,Nx,Ny,dx,eta,sim_type)
        u_du[0] = dot(uu[css],du_x)
        u_du[1] = dot(uu[css],du_y)
        lap_u[0] = lap(uu[0],dx)
        lap_u[1] = lap(uu[1],dx)

        #calculate u_stars
        u_star[css] = uu[css] +  dt*(-u_du +  eta* lap_u + f[css])
        BC_zeros(u_star[0],sim_type)
        BC_zeros(u_star[1],sim_type)

        rhs =  dx * dx * (rho/ dt) * div(u_star[0],u_star[1],dx)

        #pressure convergence loop
        for _ in range(int(50)):
            prs[cs] = (prs[2:,1:-1] +  prs[0:-2,1:-1] +  prs[1:-1,2:] +  prs[1:-1,0:-2] - rhs)/4
            BC_standard(prs,sim_type)

        #update velocities
        dev_prs[0] = dev_x(prs,dx)
        dev_prs[1] = dev_y(prs,dx)
        uu[css] =  u_star[css] - (dt/rho)*dev_prs
        BC_zeros(uu[0],sim_type)
        BC_zeros(uu[1],sim_type)


    #region running loop
    #run entire sim
    for i in range(Nt + 1):
        update_all(phi, c, p, uu, mu_phi, mu_c, h, omega, sigma_xx, sigma_xy,sigma_yx,sigma_yy, D_x,D_y, prs,
                  u_star, f, J_c, dphi ,dc, du_x, du_y, dmu_c, dmu_phi, dh_dxs, dh_dys, dp_x, dp_y, lap_u, dev_prs,
                     c_p, c_h, p_D, p_dh, dsigma, u_dp, u_du, dx, dt, M, beta, kappa, eta, rho, xl, kBT, B, gamma_r, gamma_t, sim_type)

        #save files, continues to use the same folder when loading
        if i % save == 0:
            # uncomment for conservation check
            # print(np.sum(self.c[cs]*self.dx*self.dx))  #conservation check
            # print(np.sum(self.phi[cs]*self.dx*self.dx))  #conservation check
            np.savez(f'./{name}_data/step_{i+load_no}.npz', phi=phi, c=c, px=p[0], py=p[1])
    #endregion


make_folder(name="parallel speed test",load=False)
main(name="parallel speed test",Nx=500,Ny=250,dt=0.001,dx=1.0,Nt=int(1e4),sim_type=1)

profiler.disable()
# Save the results to a file
with open("profile_output.txt", "w") as f:
    stats = pstats.Stats(profiler, stream=f)
    stats.sort_stats("cumulative")  # Sort by cumulative time or "time" to sort by time spent in the function
    stats.print_stats()

