"""
img2paint.py: Convert a standard image into a painting.

author: Frank Liu - frank.zijie@gmail.com
last modified: 07/29/2015

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
from skimage.color import label2rgb
from skimage.segmentation import slic

# @TODO: this is a hack to enable the import of RAG code in future skimage versions
# should be changed when all of the RAG-related code is no longer experimental
try:
    from skimage.graph import cut_threshold, rag_mean_color
except ImportError:
    from skimage.future.graph import cut_threshold, rag_mean_color

# argparse
parser = argparse.ArgumentParser()
parser.add_argument("-i", "--image", type=str, required=True, help="input image path")
parser.add_argument("-o", "--output", type=str, required=False, help="output path")

# expperimentally determined region adjacency threshold
RAG_THRESHOLD = 12

# SLIC compactness
SLIC_COMPACTNESS = 12

# number of desired SLIC regions
N_SLIC_REGIONS = 1024


def img2paint(img):
    """
        Turn a standard image into a painting.

        :param numpy.ndarray img:
            An RGB image in numpy format.
    """

    # scale the image
    img = cv2.convertScaleAbs(img)

    # perform SLIC segmentation
    regions_slic = slic(img, compactness=SLIC_COMPACTNESS, n_segments=N_SLIC_REGIONS)

    # cluster all SLIC regions
    rag = rag_mean_color(img, regions_slic)
    regions_rag = cut_threshold(regions_slic, rag, RAG_THRESHOLD)

    # final painting
    result = label2rgb(regions_rag, img, kind="avg")

    return result


if __name__ == "__main__":

    args = parser.parse_args()

    # perform the conversion
    img = cv2.cvtColor(cv2.imread(args.image), cv2.COLOR_BGR2RGB)
    painting = cv2.cvtColor(img2paint(img), cv2.COLOR_RGB2BGR)

    if args.output == None:
        cv2.imshow("Result", painting)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.output, painting)

