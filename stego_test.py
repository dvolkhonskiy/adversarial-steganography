import numpy as np
from image_utils.time_utils import Timer

from steganography.lsb_matching import LSBMatching

def test_lsb_matching(algorithm, iterations=100):
    with Timer():
        for i in range(iterations):
            algorithm.encode(container, information, stego)

container = 'cat.png'
information = np.random.randint(0, 2, 100)
print(information)
stego = 'stego.png'


# test_lsb_matching(LSBMatching, iterations=100)
LSBMatching.encode(container, information, stego)
# print(LSBMatching.decode(stego))
