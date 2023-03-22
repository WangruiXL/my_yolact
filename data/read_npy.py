import numpy as np

# data = np.zeros((3,4))
# np.save('data.npy', data)
# del data
data = np.load('grid.npy')

print('type :', type(data))
print('shape :', data.shape)
print('data :')
print(data)
