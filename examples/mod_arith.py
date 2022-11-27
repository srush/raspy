import math

import numpy as np
from raspy import *

V = np.arange(13)
N = 13
K = 2
w = K * (2 * math.pi) / N

reverse = select(indices, indices, neq)


def argmax(v):
    f = lambda i: v[i]
    return max(range(len(v)), key=f)


def dft_out(c, s):
    return c * np.cos(w * V) + s * np.sin(w * V)


# Map inputs x,y → cos(wx), cos(wy),sin(wx),sin(wy) with a Discrete Fourier Transform, for some frequency w
# Multiply and rearrange to get cos(w(x+y))=cos(wx)cos(wy)−sin(wx)sin(wy) and sin(w(x+y))=cos(wx)sin(wy)+sin(wx)cos(wy). By choosing a frequency w =2πnk  we get period dividing n, so this is a function of x + y(mod n )
# Map to the output logits z with cos(w(x+y))cos(wz)+sin(w(x+y))sin(wz)=cos(w(x+y−z)) - this has the highest logit at z= x+y(modn), so softmax gives the right answer.


def algorithm():
    xy = tokens
    yx = aggregate(reverse, xy)
    cos_x, sin_x = (xy * w).cos(), (xy * w).sin()
    cos_y, sin_y = (yx * w).cos(), (yx * w).sin()
    cos_xy = cos_x * cos_y - sin_x * sin_y
    sin_xy = cos_x * sin_y + sin_x * cos_y
    out = SOp.zip(dft_out, cos_xy, sin_xy)
    out = out.map(argmax)
    out = aggregate(select(indices, 0, eq), out)
    return out


x = algorithm()([5, 10])
print(x)
