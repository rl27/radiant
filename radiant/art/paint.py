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


def paint(img):
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
    painting = cv2.cvtColor(paint(img), cv2.COLOR_RGB2BGR)

    if args.output == None:
        cv2.imshow("Result", painting)
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.output, painting)

