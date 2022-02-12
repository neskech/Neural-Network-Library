import numpy as np

list = [1,2,3,4,5]
array = np.array(list).reshape( (len(list), 1) )
print( array * 5 )