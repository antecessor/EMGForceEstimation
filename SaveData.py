from PIL import Image
import os


def saveDataAsImage(sigs, firings):
    if not os.path.exists("dataA"):
        os.mkdir("dataA")
    if not os.path.exists("dataB"):
        os.mkdir("dataB")
    for idx, signal in enumerate(sigs):
        im = Image.fromarray(signal)
        an = Image.fromarray(firings[idx])
        im.save("dataA/image{0}.png".format(idx))
        an.save("dataB/image{0}.png".format(idx))
