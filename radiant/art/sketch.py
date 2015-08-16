import argparse

# library imports
import cv2
import numpy as np

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True, help="input image path")
parser.add_argument("-o", "--output", type=str, required=False, help="output path")


# alpha value to use in dodge effect
EXPOSURE_ALPHA = 1


def sketch(img):
    """
        Turn a standard image into a sketch.

        :param numpy.ndarray img:
            An RGB image in numpy format.
    """

    # scale the image
    img = cv2.convertScaleAbs(img)
    
    # grayscale and blur-inverted
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_inv = cv2.GaussianBlur(255 - img_gray, (31, 31), 0)

    # sketch-ify the image by increasing exposure
    img_gray = img_gray.astype(np.uint32)
    sketch = (img_gray*255)/(256-img_inv*EXPOSURE_ALPHA)
    sketch[np.where(sketch > 255)] = 255

    return sketch.astype(np.uint8)


if __name__ == "__main__":

    args = parser.parse_args()

    # turn the image into a sketch
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    sketch = sketch(img)

    if args.output == None:
        cv2.imshow("Result", sketch)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.output, sketch)
