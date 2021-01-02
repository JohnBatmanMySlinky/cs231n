import numpy as np

# so say we had the follow array X
x = np.arange(12).reshape(3,4)

# and I wanted to index each row of x by an array y
# essentially select the diagonals
y = np.arange(3)

# why does this work
x[np.arange(3),y]

# and this don't
x[:,y]
