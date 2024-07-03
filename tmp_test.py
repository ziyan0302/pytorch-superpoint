import pdb

if (0):
    class Test_fun(object):

        default_config = 100

        def __init__(self):
            self.config = self.default_config
        def printConfig(self):
            print(self.config)


    class Child_fun(Test_fun):

        default_config = 200

        def __init__(self):
            self.config = self.default_config



    child_fun = Child_fun()
    child_fun.config
    child_fun.config = 300

import time 
import numpy as np
t1 = time.time()
for i in range(1000):
    # tmp = list(range(10000))
    array_with_step = np.arange(0, 1, 2)
t2 = time.time()
print(t2-t1)
print(array_with_step)
# pdb.set_trace()