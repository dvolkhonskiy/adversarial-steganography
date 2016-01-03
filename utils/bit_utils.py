import binascii


# TEXT #
def text_to_bit(text):
    return bin(int(binascii.hexlify(text), 16))[2:]


def bit_to_text(bit_text):
    pass


# PNG IMAGES #
def png_to_bit(pil_img):
    pass


def bit_to_png(bit_pil_img):
    pass
