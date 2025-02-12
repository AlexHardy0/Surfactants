import numpy as np
import matplotlib.pyplot as plt
import os
import gc

class Surfactant:
	def __init__(self,
		dt=0.01, dx=1.0, Nx=64, Ny=64, Nt=1000000, save=10000, 
		M=3.0, beta=2.0, kappa=1.0, xl=0.1, kBT=1.0,
		B=0.5, gamma_r=0.01, gamma_t=0.01, eta=1.0, Np=1000,
		name='array-droplets', load=False, load_no=0, walls=1):
		
		''' 
		line 1: resolution parameters: time step, spatial res., x-lattice size, y-lattice size, total no of time steps
		line 2: equation parameters: M, beta, kappa, xl, kBT
		line 3: continued equation parameters: B, gamma_r, gamma_t, eta and Np, the number of particles to use
		line 4: name of the simulation folder to save to, loading information, and type of simulation run
		'''
		
		# set up folder to save data to
		self.name = name
		self.walls = walls
		self.crash = False
		self.load_no = load_no

		try:
			os.mkdir('./' + name + '-data')
		except OSError:
			print("folder name in use")
			if load == False:
				self.crash = True

		self.Np, self.Nt, self.Nx, self.Ny = Np, Nt, Nx, Ny
		self.dt, self.dx = dt, dx  # assuming dx == dy
		self.save = save
		self.Lx, self.Ly = Nx*dx, Ny*dx  # why integer
		self.M, self.beta, self.kappa, self.B, self.kBT = M, beta, kappa, B, kBT
		self.xl, self.eta, self.gamma_t, self.gamma_r = xl, eta, gamma_t, gamma_r

		# meshgrid for plotting
		x = np.arange(0.0,self.Lx,self.dx)
		y = np.arange(0.0,self.Ly,self.dx)
		self.Y,self.X = np.meshgrid(y,x)

		# set up fourier space parameters, based on if walls are present or not
		if self.walls:
			kx = np.fft.fftfreq(2*Nx,d=dx) * 2 * np.pi
		else:
			kx = np.fft.fftfreq(Nx,d=dx) * 2 * np.pi
		ky = np.fft.fftfreq(Ny,d=dx) * 2 * np.pi
		self.KY,self.KX = np.meshgrid(ky,kx)
		self.k2 = self.KX**2 + self.KY**2
		self.k4 = self.k2 * self.k2
	
		# scalar field array definitions
		self.phi = np.zeros((Nx,Ny))
		self.mu_phi = np.zeros((Nx,Ny))
		self.c = np.zeros((Nx,Ny))
		self.mu_c = np.zeros((Nx,Ny))
		self.dmu_c_dx = np.zeros((Nx,Ny))
		self.dmu_c_dy = np.zeros((Nx,Ny))

		# vector field array definitions
		self.u_x = np.zeros((Nx,Ny))
		self.u_y = np.zeros((Nx,Ny))
		if self.walls:
			self.u_x_k = np.zeros((2*Nx,Ny), dtype=complex)
			self.u_y_k = np.zeros((2*Nx,Ny), dtype=complex)
		else:
			self.u_x_k = np.zeros((Nx,Ny), dtype=complex)
			self.u_y_k = np.zeros((Nx,Ny), dtype=complex)
		self.p_x = np.zeros((Nx,Ny))
		self.p_y = np.zeros((Nx,Ny))
		self.c_p_x = np.zeros((Nx,Ny))
		self.c_p_y = np.zeros((Nx,Ny))
		self.f_x = np.zeros((Nx,Ny))
		self.f_y = np.zeros((Nx,Ny))
		self.h_x = np.zeros((Nx,Ny))
		self.h_y = np.zeros((Nx,Ny))

		# matrix array defintions
		self.Omega_xy = np.zeros((Nx,Ny))
		self.Omega_yx = np.zeros((Nx,Ny))
		self.D_xx = np.zeros((Nx,Ny))
		self.D_xy = np.zeros((Nx,Ny))  
		self.D_yy = np.zeros((Nx,Ny))
		self.sigma_xx = np.zeros((Nx,Ny))
		self.sigma_xy = np.zeros((Nx,Ny))
		self.sigma_yx = np.zeros((Nx,Ny))
		self.sigma_yy = np.zeros((Nx,Ny))

		# intialize phi and c, based on simulation type + if you are loading 
		if load == False:
			# initalise phi
			if self.walls:
				self.init_phi_flat()
			else:
				self.init_phi_many()
			# initialise c
			self.init_c()
		else:
			#load from folder
			self.phi = np.loadtxt(f'./{name}-data/phi{load_no}.txt')
			self.c = np.loadtxt(f'./{name}-data/c{load_no}.txt')
			self.px = np.loadtxt(f'./{name}-data/px{load_no}.txt')
			self.py = np.loadtxt(f'./{name}-data/py{load_no}.txt')

	def init_phi_many(self):
		
		'''	initialize phi with multiple small drops '''
		self.phi[:, :] = -1.0
		
		# small drop parameters
		size = self.Nx
		center = size // 2
		off_center = size // 4
		if self.Nx == 64:
			radius = 9
		else:
			radius = 18

		# Function to place a circle in phi
		def place_circle(xc, yc, r):
			'''	function for placing small drops in phi at (xc, yc) '''
			for x in range(size):
				for y in range(size):
					if (x - xc) ** 2 + (y - yc) ** 2 < r ** 2:
						self.phi[x, y] = 1.0

		# Place the corner drop
		corner_positions = [(0, 0), (0, size), (size, 0), (size, size)]
		for (xc, yc) in corner_positions:
			place_circle(xc, yc, radius)

		# Place the top and bottom drops
		place_circle(center, 0, radius)       
		place_circle(center, size, radius) 

		# Place the left and right drops
		place_circle(0, center, radius,)      
		place_circle(size, center, radius)   

		# Place drop in the centre
		place_circle(center+1, center+1, radius)

		# Place dropin the gaps between
		place_circle(off_center-1,off_center+1,radius)
		place_circle(off_center+center+1,off_center+1,radius)
		place_circle(off_center-1,off_center+center-1,radius)
		place_circle(off_center+center,off_center+center,radius)

		# plt.pcolormesh(self.X,self.Y,self.phi)
		# plt.show()

	def init_phi_flat(self):
		'''	initalize phi array as a flat vertical interface halfway '''
		print("half")
		half = int(self.Nx/2)
		self.phi[:,:] = 1.0
		self.phi[:half,:] = -1.0

	def init_c(self):
		'''	initalise c as uniform everywhere '''
		self.c[:,:] = self.Np/(self.Lx*self.Ly)

	def dev_x(self, q, factor):
		''' derivative in x direction, q: array to differentiate '''
		# factor is either -1, 1, or zero
		# make padded array to add extra layers on each 4 sides of q
		Q = np.zeros((self.Nx+2, self.Ny+2))
		Q[1:-1,1:-1] = q
		Q[1:-1,0] = q[:,-1]  # q(j=-1) = q(j=Ny-1)  periodic boundary condition at y=0 and y=Ly
		Q[1:-1,-1] = q[:,0]  # q(j=Ny) = q(j=0)
		if self.walls:
			Q[0,:] = Q[1,:]*factor   # q(i=-1) = q(i=0)  wall boundary condition at x=0 and x=Lx
			Q[-1,:] = Q[-2,:]*factor # q(i=Ny) = q(i=Ny-1)  give negative vector for J
		else:
			Q[0,1:-1] = q[-1,:]
			Q[-1,1:-1] = q[0,:]

		return (Q[2:,1:-1] - Q[0:-2,1:-1])/(2*self.dx)

	def dev_y(self, q, factor):
		''' derivative in y direction, q: array to differentiate '''
		# factor is either -1, 1, or zero
		# make padded array to add extra layers on each 4 sides of q
		Q = np.zeros((self.Nx+2, self.Ny+2))
		Q[1:-1,1:-1] = q
		Q[1:-1,0] = q[:,-1]  # q(j=-1) = q(j=Ny-1)  periodic boundary condition at y=0 and y=Ly
		Q[1:-1,-1] = q[:,0]  # q(j=Ny) = q(j=0)
		if self.walls:
			Q[0,:] = Q[1,:]*factor   # q(i=-1) = q(i=0)  wall boundary condition at x=0 and x=Lx
			Q[-1,:] = Q[-2,:]*factor # q(i=Ny) = q(i=Ny-1)  give negative vector for J
		else:
			Q[0,1:-1] = q[-1,:]
			Q[-1,1:-1] = q[0,:]

		return (Q[1:-1,2:] - Q[1:-1,0:-2])/(2*self.dx)
	
	def lap(self, q, factor):
		''' calculates grad^2(q) '''
		# factor is either -1, 1, or zero
		# make padded array to add extra layers on each 4 sides of q
		Q = np.zeros((self.Nx+2, self.Ny+2))
		Q[1:-1,1:-1] = q
		Q[1:-1,0] = q[:,-1]  # q(j=-1) = q(j=Ny-1)  periodic boundary condition at y=0 and y=Ly
		Q[1:-1,-1] = q[:,0]  # q(j=Ny) = q(j=0)
		if self.walls:
			Q[0,:] = Q[1,:]*factor   # q(i=-1) = q(i=0)  wall boundary condition at x=0 and x=Lx
			Q[-1,:] = Q[-2,:]*factor # q(i=Ny) = q(i=Ny-1)  give negative vector for J
		else:
			Q[0,1:-1] = q[-1,:]
			Q[-1,1:-1] = q[0,:]

		dqdx2 = (Q[2:,1:-1] - 2*Q[1:-1,1:-1] + Q[0:-2,1:-1])/(self.dx*self.dx)
		dqdy2 = (Q[1:-1,2:] - 2*Q[1:-1,1:-1] + Q[1:-1,0:-2])/(self.dx*self.dx)

		return dqdx2 + dqdy2

	def div(self, qx, qy, factor):
		''' calculates divergence of a vector field (nabla . q) '''
		dqx_dx = self.dev_x(qx, factor)
		dqy_dy = self.dev_y(qy, factor)

		return dqx_dx + dqy_dy

	def grad(self, q, factor):
		''' calculates the gradient of a scalar field (grad(q)) '''
		dq_dx = self.dev_x(q, factor)
		dq_dy = self.dev_y(q, factor)

		return [dq_dx, dq_dy]

	def update_all(self):
		'''	update only the 'real' values for each field '''
		# calculate commonly used gradient
		dphi_dx, dphi_dy = self.grad(self.phi, 1)
		du_x_dx, du_x_dy = self.grad(self.u_x, 1)
		du_y_dx, du_y_dy = self.grad(self.u_y, 1)
		dc_dx, dc_dy = self.grad(self.c, 1)

		##############################################################################
		# PRELIMINARY CALCULATIONS FOR UPDATED VARIABLES + APPLY APPROPIATE BC'S
		##############################################################################
		#calculate mu_phi
		self.c_p_x = self.c*self.p_x
		self.c_p_y = self.c*self.p_y

		self.mu_phi = self.beta*(self.phi**3 - self.phi) - self.kappa*self.lap(self.phi, 1) \
					- self.xl*self.div(self.c_p_x,self.c_p_y, 1)

		# calculate 2/3 of mu_c
		self.mu_c = self.kBT*(self.p_x**2 + self.p_y**2) + self.xl*(self.p_x*dphi_dx + self.p_y*dphi_dy)

		# calculate h (without c)
		self.h_x = 2*self.kBT*self.p_x + self.xl*dphi_dx
		self.h_y = 2*self.kBT*self.p_y + self.xl*dphi_dy

		# calculate Omega
		self.Omega_xy = 0.5*(du_y_dx - du_x_dy)
		self.Omega_yx = 0.5*(du_x_dy - du_y_dx)

		# calculate D
		self.D_xx = du_x_dx
		self.D_xy = 0.5*(du_y_dx + du_x_dy)
		self.D_yy = du_y_dy

		# calculate sigma
		c_h_x = self.c*self.h_x
		c_h_y = self.c*self.h_y

		self.sigma_xx = (self.B/2) * (self.p_x*c_h_x)
		self.sigma_xy = (self.p_x*c_h_y)*((self.B/4)-0.5) + (self.p_y*c_h_x)*((self.B/4)+0.5)
		self.sigma_yx = (self.p_y*c_h_x)*((self.B/4)-0.5) + (self.p_x*c_h_y)*((self.B/4)+0.5)
		self.sigma_yy = (self.B/2) * (self.p_y*c_h_y)

		# calculate f
		dh_x_dx, dh_x_dy = self.grad(c_h_x, 1)
		dh_y_dx, dh_y_dy = self.grad(c_h_y, 1)

		self.f_x = -self.phi*self.dev_x(self.mu_phi,1) - self.c*self.dev_x(self.mu_c,1) \
				 - self.p_x*dh_x_dx - self.p_y*dh_y_dx \
				 - self.kBT*dc_dx + self.dev_x(self.sigma_xx,1) + self.dev_y(self.sigma_xy,1)  # mistake here
		self.f_y = -self.phi*self.dev_y(self.mu_phi,1) - self.c*self.dev_y(self.mu_c,1) \
				 - self.p_x*dh_x_dy - self.p_y*dh_y_dy \
				 - self.kBT*dc_dy + self.dev_x(self.sigma_yx,1) + self.dev_y(self.sigma_yy,1)

		##################################################################################
		# UPDATE ALL FIELDS + APPLY APPROPIATE BC'S
		##################################################################################
		phi_u_x = self.phi*self.u_x
		phi_u_y = self.phi*self.u_y
		c_u_x = self.c*self.u_x
		c_u_y = self.c*self.u_y

		# update phi
		self.phi += self.dt*(-self.div(phi_u_x, phi_u_y, 1) + self.M*self.lap(self.mu_phi, 1)) 

		# update c
		self.dmu_c_dx,self.dmu_c_dy = (self.c/self.gamma_t)*self.grad(self.mu_c, 1)
		self.c += self.dt*(-self.div(c_u_x, c_u_y, -1)+ (self.kBT/self.gamma_t)*self.lap(self.c, 1) \
						  + self.div(self.dmu_c_dx, self.dmu_c_dy, -1))    

		# update px
		self.p_x += self.dt*(-self.u_x*self.dev_x(self.p_x, 1) - self.u_y*self.dev_y(self.p_x, 1) \
							+ self.Omega_xy*self.p_y \
							- (self.B/2)*(self.D_xx*self.p_x + self.D_xy*self.p_y) \
							- (1/(2*self.gamma_r))*self.h_x)

		# update py
		self.p_y += self.dt*(-self.u_x*self.dev_x(self.p_y, 1) - self.u_y*self.dev_y(self.p_y, 1) \
							+ self.Omega_yx*self.p_x \
							- (self.B/2)*(self.D_xy*self.p_x + self.D_yy*self.p_y) \
							- (1/(2*self.gamma_r))*self.h_y)

		# create flipped f (if walls are on)
		if self.walls:
			fx = np.zeros((2*self.Nx,self.Ny))
			fx[:self.Nx,:] = self.f_x
			fx[self.Nx:,:] = -self.f_x[::-1,:]

			fy = np.zeros((2*self.Nx,self.Ny))
			fy[:self.Nx,:] = self.f_y
			fy[self.Nx:,:] = -self.f_y[::-1,:]
		else:
			fx = self.f_x
			fy = self.f_y

		# fourier parameters
		fx_k = np.fft.fft2(fx,norm='ortho')*self.dx
		fy_k = np.fft.fft2(fy,norm='ortho')*self.dx

		# preliminary calculations for velocity
		k_f = (self.KX*fx_k) + (self.KY*fy_k)

		# to avoid a divide by 0
		ind = (self.k2 != 0)
		# index that is dodged remains zeros anyways (i think)

		# velocity calculations
		self.u_x_k[ind] = ((fx_k[ind]/self.k2[ind]) - (self.KX[ind]*k_f[ind]/self.k4[ind]))/self.eta
		self.u_y_k[ind] = ((fy_k[ind]/self.k2[ind]) - (self.KY[ind]*k_f[ind]/self.k4[ind]))/self.eta

		# transform u back into real space
		self.u_x = (np.fft.ifft2(self.u_x_k,norm='ortho').real/self.dx)[:self.Nx,:]
		self.u_y = (np.fft.ifft2(self.u_y_k,norm='ortho').real/self.dx)[:self.Nx,:]

		mag = np.hypot(self.u_x,self.u_y)
		plot = plt.quiver(self.X,self.Y,self.u_x,self.u_y,mag)
		plt.colorbar(plot)
		plt.show()

	def run(self):
		''' loop for entire simulation '''
		for n in range(self.Nt + 1):
			self.update_all()

			# save files, continues to use the same folder when loading
			if n % self.save == 0:
				print(f'n = {n}')
				print(f'Np = {np.sum(self.c*self.dx*self.dx)}')  # conservation check
				print(f'phi_avg = {np.sum(self.phi*self.dx*self.dx)/(self.Lx*self.Ly)}')  # conservation check
				np.savetxt(f'./{self.name}-data/phi{n+self.load_no}.txt', self.phi)
				np.savetxt(f'./{self.name}-data/c{n+self.load_no}.txt', self.c)
				np.savetxt(f'./{self.name}-data/px{n+self.load_no}.txt', self.p_x)
				np.savetxt(f'./{self.name}-data/py{n+self.load_no}.txt', self.p_y)
				np.savetxt(f'./{self.name}-data/ux{n+self.load_no}.txt', self.u_x)
				np.savetxt(f'./{self.name}-data/uy{n+self.load_no}.txt', self.u_y)

def run_simulation(simulation_class, *args, **kwargs):
	''' 
	simulation_class: Class of the simulation
	args: Positional arguments for the simulation class.
	kwargs: Keyword arguments for the simulation class.
	'''
	try:
		sim = simulation_class(*args, **kwargs)
		# only run when passed the initialization
		if not sim.crash:  
			sim.run()  
		print(f"{kwargs.get('name', 'Simulation')} passed")
	except Exception as e:
		print(f"{kwargs.get('name', 'Simulation')} crashed: {e}")
	finally:
		# explicitly delete the object and force garbage collection
		del sim
		gc.collect()


# run_simulation(Surfactant, name='interface-no-surfactant',Nt=3000000, xl=0.0, Np=0,   walls=1, dt = 0.0001, dx=0.25, Nx=256, Ny=8)

# run_simulation(Surfactant, name='interface-surfactant-3-np-400', Nt=3000000, xl=0.3, Np=400, walls=1, dt = 0.0001, dx=0.25, Nx=256, Ny=8,load=True,load_no=int(6e6))
# run_simulation(Surfactant, name='interface-surfactant-5-np-400', Nt=3000000, xl=0.5, Np=400, walls=1, dt = 0.0001, dx=0.25, Nx=256, Ny=8,load=True,load_no=int(6e6))

# run_simulation(Surfactant, name='interface-surfactant-3-np-800', Nt=3000000, xl=0.3, Np=800, walls=1, dt = 0.0001, dx=0.25, Nx=256, Ny=8,load=True,load_no=int(6e6))
# run_simulation(Surfactant, name='interface-surfactant-5-np-800', Nt=3000000, xl=0.5, Np=800, walls=1, dt = 0.0001, dx=0.25, Nx=256, Ny=8,load=True,load_no=int(6e6))

# run_simulation(Surfactant, name='interface-surfactant-3-np-200', Nt=3000000, xl=0.3, Np=200, walls=1, dt = 0.0001, dx=0.25, Nx=256, Ny=8,load=True,load_no=int(3e6))
# run_simulation(Surfactant, name='interface-surfactant-5-np-200', Nt=3000000, xl=0.5, Np=200, walls=1, dt = 0.0001, dx=0.25, Nx=256, Ny=8,load=True,load_no=int(2e6))

# run_simulation(Surfactant, name='emulsion-surfactant-small-vrs-more', Nt=int(10e6), xl=1.5, Np=1000, walls=0, dt = 0.001, dx=1.0, Nx=64, Ny=64)
# run_simulation(Surfactant, name='emulsion-no-surfactant', Nt=6000000, xl=0.0, Np=0,    walls=0, dt = 0.001, dx=1.0, Nx=128, Ny=128,load=True,load_no=int(6e6))
# run_simulation(Surfactant, name='emulsion-surfactant',    Nt=6000000, xl=1.5, Np=1000, walls=0, dt = 0.001, dx=1.0, Nx=128, Ny=128,load=True,load_no=int(6e6))

# x = np.arange(0.0,64,1)
# y = np.arange(0.0,64,1)
# Y,X = np.meshgrid(y,x)

# plt.pcolormesh(X,Y,np.loadtxt(f"./emulsion-surfactant-small-vrs-more-data/phi{int(10e6)}.txt"))
# plt.show()

run_simulation(Surfactant,name='test',walls=0,Nt=int(0))