import numpy as np
import scipy as sp
from sklearn import preprocessing 

a = np.array([[1,2,3,4],[1,2,3,4]])
print(a)
b = sp.amax(a)
print(b)
c = preprocessing.data.check_array(a)
print(c)