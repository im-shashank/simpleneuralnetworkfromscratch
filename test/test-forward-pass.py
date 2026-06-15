import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main.MLP.mlp import *

x = [2.0, 3.0, -1.0]
m = mlp(len(x), [4, 4, 1])
o = m(x)
print([d.data for d in o])