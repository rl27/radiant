import argparse
import multiprocessing
import sys
import timeit

import cv2
import numpy as np
import weave

# face cascades
FRONTAL_FACE_CASCADE_FNAME = "haarcascade_frontalface_default.xml"
PROFILE_FACE_CASCADE_FNAME = "haarcascade_profileface.xml"
UPPER_BODY_CASCADE_FNAME = "haarcascade_upperbody.xml"
FULL_BODY_CASCADE_FNAME = "haarcascade_fullbody.xml"

# default index type
INDEX_DTYPE = np.uint32

# allowed extensions
ALLOWED_EXTENSIONS = ["jpg", "jpeg", "png"]

# argparse
parser = argparse.ArgumentParser(description="Seam carving for content-aware image resizing.",
                                 usage="seam.py -i <image_path> -d <new_width> <new_height>")
parser.add_argument("-i", "--image", type=str, required=True, help="input image to resize")
parser.add_argument("-d", "--dims", nargs=2, type=int, required=True, help="output dimensions")
parser.add_argument("-o", "--output", type=str, required=False, help="output path")
parser.add_argument("-m", "--mark", action="store_true", required=False, help="mark seams")

# multiprocessing
pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())


def _checkExtension(path):
    """
        Ensures that the path is loadable by OpenCV.
    """

    ext = path.split(".")[-1]
    if ext.lower() in ALLOWED_EXTENSIONS:
        return True
    return False

def _checkDimensions(img, new_dims):
    """
        Validates the new image dimensions.
    """

    assert new_dims[0] > 0 and new_dims[1] > 0, "new width and height must be positive integers"
    assert new_dims[0] <= img.shape[1]*2, "new width cannot exceed 2x the original's"
    assert new_dims[1] <= img.shape[0]*2, "new height cannot exceed 2x the original's"

def detectFaces(img_gray):
    """
        Detects frontal and profile faces in an image.
    """

    # thank you, OpenCV
    vj_detector_frontal = cv2.CascadeClassifier(FRONTAL_FACE_CASCADE_FNAME)
    vj_detector_profile = cv2.CascadeClassifier(PROFILE_FACE_CASCADE_FNAME)
    regions = []
    regions.extend(vj_detector_frontal.detectMultiScale(img_gray,
                                                        scaleFactor=1.2,
                                                        minNeighbors=5))
    regions.extend(vj_detector_profile.detectMultiScale(img_gray,
                                                        scaleFactor=1.2,
                                                        minNeighbors=5))
    
    return regions

def getEnergyImage(img_gray, protect=None):
    """
        Returns the energy image of the input.
    """

    # image derivative (x+y of Sobel)
    img_en_x = cv2.Sobel(img_gray, cv2.CV_32F, 1, 0, ksize=3)
    img_en_y = cv2.Sobel(img_gray, cv2.CV_32F, 0, 1, ksize=3)
    img_en = np.fabs(img_en_x) + np.fabs(img_en_y)

    # regions to protect
    if protect is not None:
        img_en[np.where(protect != 0)] = np.inf

    return img_en

def extractSeam(img_en, use_weave=True):
    """
        Extracts a single vertical seam, given an energy image.
    """

    # instantiate cumulative energy map and back pointers
    inf_list = [np.inf] * img_en.shape[0]
    inf_col = np.array([inf_list], dtype=img_en.dtype).T
    cum_en = np.hstack((inf_col, img_en, inf_col))
    back = np.zeros(img_en.shape, dtype=np.uint8)

    # propagate energies (fast)
    if use_weave:
        code = """
            int pos = 0;
            for(int i = 1; i < Nimg_en[0]; i++) {
                for(int j = 0; j < Nimg_en[1]; j++) {
                    double min = cum_en[pos],
                           v1 = cum_en[pos+1],
                           v2 = cum_en[pos+2];
                    int idx = 0;

                    // get the minimum value and its index
                    if(v1 < min) {
                        min = v1;
                        idx = 1;
                    }
                    if(v2 < min) {
                        min = v2;
                        idx = 2;
                    }

                    // fill the cumulative energy and back pointers
                    cum_en[pos+Ncum_en[1]+1] = img_en[i*Nimg_en[1]+j] + min;
                    back[i*Nimg_en[1]+j] = idx;
                    pos++;
                }
                pos+=2;
            }
        """

        weave.inline(code, ["img_en", "cum_en", "back"])

    # propagate energies (slow)
    else:
        for y in range(1, img_en.shape[0]):
            for x in range(img_en.shape[1]):

                # compute optimal next pixel by row
                options = cum_en[y-1,x:x+3]
                idx = options.argmin()
                
                # fill the cumulative energy image and back pointers
                cum_en[y,x+1] = img_en[y,x] + options[idx]
                back[y,x] = idx

    # extract n_seam seams
    seam = []
    col = cum_en[-1,1:-1].argmin()
    for j in range(img_en.shape[0]-1, 0, -1):
        seam.append(col)
        col += back[j,col] - 1
    seam.append(col)

    return np.array(seam[::-1], dtype=INDEX_DTYPE)

def removeSeam(img, seam):
    """
        Removes a vertical seam from an input image.
    """

    # create a new image with the seam removed
    new_shape = list(img.shape)
    new_shape[1] -= 1
    img_new = np.zeros(new_shape, dtype=img.dtype)
    for i in range(img.shape[0]):
        img_new[i,0:seam[i]] = img[i,0:seam[i]]
        img_new[i,seam[i]:] = img[i,seam[i]+1:]

    return img_new

def enlargeImage(img, seams):
    """
        Enlarges an image given an array of seams.
    """

    # create a new image
    new_shape = list(img.shape)
    new_shape[1] += len(seams)
    img_new = np.zeros(new_shape, dtype=img.dtype)

    # sort the seams by index to get splits
    seams.sort(axis=0)
    for i in range(img.shape[0]):
        split = seams[:,i]

        # enlarge the image row-by-row via splits
        img_new[i,0:split[0]] = img[i,0:split[0]]
        for j in range(split.size):
            img_new[i,split[j-1]+j:split[j]+j] = img[i,split[j-1]:split[j]]
            img_new[i,split[j]+j] = img[i,split[j]]
        img_new[i,split[-1]+split.size:] = img[i,split[-1]:]
    
    return img_new

def markSeams(img, seams, color=(255, 0, 0)):
    """
        Marks a set of extracted scenes.
    """

    # mark the seams
    xi = np.arange(img.shape[0])
    for seam in seams:
        img[xi,seam] = color

def content_aware_resize(img, new_size, do_mark=False, do_yield_subimages=False):
    """
        Performs a smart resize operation. Entry point for seam carving.

        :param numpy.ndarray img:
            An RGB image in numpy format.

        :param tuple new_size:
            Image resize parameters (width, height). 
    """

    # validate image dimensions
    _checkDimensions(img, new_size)

    start = timeit.default_timer()

    # status message
    counter = 0
    stdout_orig = sys.stdout
    sys.stdout = sys.stderr
    print("Processing seam..."),

    # determine resize parameters
    seam_dims = (new_size[0] - img.shape[1], 
                 new_size[1] - img.shape[0])
    n_seams = abs(seam_dims[0]) + abs(seam_dims[1])

    # if a marked image is requested, create it
    if do_mark:
        img_mark = img.copy()

    # iteration 0: vertical seams (horizontal resize)
    # iteration 1: horizontal seams (vertical resize)
    for ns in seam_dims:
        do_store_seams = do_mark or ns > 0
        img_orig = img

        # maintain seams and an "original index" 2D array, if needed
        if do_store_seams:
            hrange = np.arange(img.shape[0])
            seams = np.zeros((0, img.shape[0]), dtype=INDEX_DTYPE)
            img_idx = np.tile(np.arange(img.shape[1], dtype=INDEX_DTYPE), (img.shape[0], 1))
        
        # loop through, removing seams for each iteration
        for n in range(abs(ns)):
            counter += 1
            update_str = "{0} of {1}".format(counter, n_seams)
            print(update_str),

            # perform seam carving operation
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img_en = getEnergyImage(img_gray)
            seam = extractSeam(img_en)
            img_next = removeSeam(img, seam)

            # store the seam, if needed
            if do_store_seams:
                seam_orig = img_idx[hrange, seam]
                img_idx = removeSeam(img_idx, seam)
                seams = np.vstack((seams, seam_orig))

            img = img_next
            print("\b"*(len(update_str)+2)),

        # mark or enlarge the image, if necessary
        if do_mark:
            img = img_orig
            markSeams(img_mark, seams)
        elif ns > 0:
            img = enlargeImage(img_orig, seams)

        # for next iteration
        img = img.transpose((1, 0, 2))
        if do_mark:
            img_mark = img_mark.transpose((1, 0, 2))

    # complete status message + timing information
    print("done.{0}".format(" "*10))
    print("Total time elapsed: {0}s".format(timeit.default_timer() - start))
    sys.stdout = stdout_orig

    return img if not do_mark else img_mark

class SmartResizer:
    """
        Wrapper class for content-aware image resizing.
    """

    def __init__(self, img):
        self.img = np.array(img)

    def resize(self, new_dims):
        """
            Resize the source image with the new dimensions.
        """

        return content_aware_resize(self.img, new_dims)

    def step_resize(self, new_dims):
        """
            Yield subimages in the seam carving process.
        """
        
        img_rsz = content_aware_resize(self.img, new_dims)
        img_rsz_seams = content_aware_resize(self.img, new_dims, do_mark=True)

        return (img_rsz, img_rsz_seams)

if __name__ == "__main__":

    args = parser.parse_args()

    # load the image
    err_msg = "must specify a valid image file"
    assert _checkExtension(args.image), err_msg
    img = cv2.imread(args.image)
    assert img is not None, err_msg

    # perform the resize operation
    dims = tuple(args.dims)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_rsz = content_aware_resize(img_rgb, dims, do_mark=args.mark)
    img_rsz = cv2.cvtColor(img_rsz, cv2.COLOR_RGB2BGR)

    # display image or write to output path
    if args.output == None or not _checkExtension(args.output):
        img_naive = cv2.resize(img, dims)
        cv2.imshow("Naive Resizing", img_naive)
        cv2.imshow("Smart Resizing", cv2.convertScaleAbs(img_rsz))
        cv2.waitKey()
        cv2.destroyAllWindows()
    else:
        cv2.imwrite(args.output, img_rsz)

    pool.close()
    pool.join()
