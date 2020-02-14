from PIL import Image
import os
import argparse


def resize(path, t=800):
    input_dir = os.listdir(path)
    for item in input_dir:
        if os.path.isfile(path+item) and item.endswith("jpg"):
            im = Image.open(path+item)
            w, h = im.size
            if w > t or h > t:
                if w >= h:
                    im = im.resize((t, int(h / (w/t))), Image.ANTIALIAS)
                else:
                    im = im.resize((int(w / (h/t)), t), Image.ANTIALIAS)
            f, e = os.path.splitext(path+item)
            im.save(f + '.jpg', 'JPEG', quality=100)


if __name__ == "__main__":
    """
    Resize all jpg images in a specific path keeping the original rateo
    Example: python resize_jpg.py -i images/ -t 800
    Effect: All images whose width or height is above 1000 will be resized: (2000, 1500) -> (1000, 750)
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str, required=True,
                    help="path to input folder containing the images")
    ap.add_argument("-t", "--threshold", type=int, default=800,
                    help="threshold above the which resize an image")
    args = vars(ap.parse_args())

    resize(args["input"], args["threshold"])

    print("Image resized")
