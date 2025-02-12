import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0,500,1.0)
y = np.arange(0,250,1.0)
Y,X = np.meshgrid(y,x)

# Load the saved .npz file
data = np.load(f'./speed test_data/step_{10000}.npz')

# Access the arrays inside the .npz file
phi_load = data['phi']
c_load = data['c']
px_load = data['px']
py_load = data['py']

fig,ax = plt.subplots()

ax.pcolormesh(X,Y,phi_load)

plt.show()