"""
img2sketch.py: Convert a standard image into a sketch.

author: Frank Liu - frank.zijie@gmail.com
last modified: 07/30/2015

Copyright (c) 2015, Frank Liu
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:
    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.
    * Neither the name of the Frank Liu (fzliu) nor the
      names of its contributors may be used to endorse or promote products
      derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL Frank Liu (fzliu) BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

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


def img2sketch(img):
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
    sketch = img2sketch(img)

    if args.output == None:
        cv2.imshow("Result", sketch)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.output, sketch)
