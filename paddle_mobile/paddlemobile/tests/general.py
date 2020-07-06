
import numpy as np
import paddlemobile as pm

arr=np.ones((6,2,4)).astype('float32') * 3
paddlebuf = pm.PaddleBuf(arr)
tensor = pm.PaddleTensor()
tensor.shape = (6,2,4)
tensor.data = paddlebuf
arr2 = np.array(tensor, copy = False)
print arr2.strides
