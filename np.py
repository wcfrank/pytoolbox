import numpy as np

# Increase the dimensions
# it is often used as transform the dimension of the image into the batch
# e.g. there is an image of shpae (height, width, channels),
# use expand_dims(image, axis=0) to make it as (1, height, width, channels)
x = np.array([1,2]) # x.shape = (2,)
y = np.expand_dims(x, axis=0) # y.shape = (1,2)

# Slice index out of range
# If the slice interval is out of range. 
# Only output the existed element without raising error
arr = np.array([1,2,3,4,5,6,7])
arr[6:10] # output: array([7])

for i in range(0,9,2):
    print(i, arr[i,i+2], end=', ') # output: 0 [1 2], 2 [3 4], 4 [5 6], 6 [7], 8 [], 
  
