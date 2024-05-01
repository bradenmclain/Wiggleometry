import numpy
from icecream import ic
from scipy.ndimage import convolve

# make some geometry with a blobby thing and a stick
thing=numpy.zeros((10,10))
thing[1:5,3:7]=1
thing[3:,5]=1
print(thing)

kernel=numpy.array([[0,0,0],[1,0,1],[0,0,0]]) # kernel is just testing for lateral expansion

test=convolve(thing,kernel)*thing # convovle the kernel and then mask the original object
print(test)
