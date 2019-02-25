import numpy as np

# increase the dimensions
x = np.array([1,2]) # x.shape = (2,)
y = np.expand_dims(x, axis=0) # y.shpae = (1,2)
# It is often used as transform the dimension of the image into the batch
# e.g. there is an image of shpae (height, width, channels),
# use expand_dims(image, axis=0) to make it as (1, height, width, channels)