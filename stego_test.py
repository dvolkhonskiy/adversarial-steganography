

from steganography.algorithms.lsb_matching import LSBMatching

container = 'cat.png'
information = '0011010'
stego = 'stego.png'

LSBMatching.encode(container, information, stego)
print(LSBMatching.decode(stego))
