import numpy as np

test = np.zeros((4,3,3))

test[0,:,:] = np.arange(0,9).reshape(3,3)


print(test)