import numpy as np

tile1 = np.tile([0, 0], (4, 1))
print(tile1)
shape1 = np.shape(tile1)
print(shape1)

tile2 = np.tile([0, 0], (4, 2))
print(tile2)
shape2 = np.shape(tile2)
print(shape2)

"""
打印结果：
        [[0 0]
         [0 0]
         [0 0]
         [0 0]]         tile1
        (4, 2)          shape1
        [[0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]
         [0 0 0 0]]     tile2
        (4, 4)          shape2
"""