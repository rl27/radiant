import argparse

import cv2
import numpy as np

# argparse
parser = argparse.ArgumentParser(description="Image deblurring via Wiener deconvolution.",
                                 usage="deconv.py -i <image_path> -k <kernel_type> -a <kernel_orientation>")
parser.add_argument("-i", "--image", type=str, required=True, help="input image path to deblur")
parser.add_argument("-t", "--type", type=str, required=False, help="kernel type")
parser.add_argument("-s", "--length", type=int, required=False, help="kernel length")
parser.add_argument("-a", "--angle", type=int, required=False, help="kernel orientation (in degrees)")
parser.add_argument("-o", "--output", type=str, required=False, help="output path")


def blur_edge(img, d):
    """
        Blur the edges of a wrapped input image.

        Taken directly from git::Itseez/opencv/samples/python2/deconvolution.py.

        :param numpy.ndarray img:
            Input image, in numpy format.

        :param int d:
            Edge length to blur by.
    """

    h, w = img.shape[:2]

    # blur the entire image, with vertical and horizontal pixel wrapping
    kernel_size = (2*d+1, 2*d+1)
    img_pad = cv2.copyMakeBorder(img, d, d, d, d, cv2.BORDER_WRAP)
    img_blur = cv2.GaussianBlur(img_pad, kernel_size, -1)[d:-d,d:-d]
    
    # map the distances to the center
    y, x = np.indices((h, w))
    dist = np.dstack([x, w-x-1, y, h-y-1]).min(-1)
    weight = np.minimum(dist.astype(np.float32)/d, 1.0)
    weight = np.repeat(weight[..., np.newaxis], 3, axis=2)
    
    return img*weight + img_blur*(1-weight)


def makeLinearKernel(flen, ksize, angle):
    """
        Creates a linear PSF. Useful for motion blur.
    """

    # create the filter
    filt = np.ones((1, flen), dtype=np.float32)

    # warp the kernel by the specified angle
    cos = np.cos(angle)
    sin = np.sin(angle)
    T = np.array([[cos, -sin, ksize/2-cos*(flen/2)], 
                  [sin,  cos, ksize/2-sin*(flen/2)]])
    kernel = cv2.warpAffine(filt, T, (ksize, ksize))
    kernel /= kernel.sum()

    return kernel


def makeCircularKernel(flen, ksize):
    """
        Creates a circular PSF. Useful for lens/focus blur.
    """

    # compute the kernel
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel = cv2.circle(kernel, (flen/2, flen/2), flen, 
                        color=1, thickness=-1, 
                        lineType=cv2.LINE_AA, shift=1)
    kernel /= kernel.sum()

    return kernel


def apply_filter(img, ftype, flen, ksize=-1, ori=0):
    """
        Apply a particular deblur filter.

        :param str ftype:
            Filter type.

        :param int ksize:
            Kernel size.

        :param int flen:
            Filter length (must be strictly smaller than the kernel size).

        :param int ori:
            Optional kernel orientation. Unused for circular kernels.
    """

    if ksize < 0:
        ksize = flen

    assert ksize >= flen, "kernel size must not be less than filter length"

    img = np.array(img, dtype=np.float32)

    # frequency domain representation of the image
    img = blur_edge(img, flen)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    IMG = cv2.dft(img_gray, flags=cv2.DFT_COMPLEX_OUTPUT)
    print(IMG)

    # make the PSF for deblurring
    if ftype == "linear":
        psf = makeLinearKernel(ksize=ksize, flen=flen, angle=ori)
    else:
        psf = makeCircularKernel(ksize=ksize, flen=flen)

    # perform the deconvolution
    psf_pad = np.zeros_like(img_gray)
    kh, kw = psf.shape
    psf_pad[:kh, :kw] = psf
    PSF = cv2.dft(psf_pad, flags=cv2.DFT_COMPLEX_OUTPUT, nonzeroRows=kh)
    PSF2 = (PSF**2).sum(-1)
    iPSF = PSF / (PSF2 + 10**5)[...,np.newaxis]
    RES = cv2.mulSpectrums(IMG, iPSF, 0)
    res = cv2.idft(RES, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
    res = np.roll(res, -kh//2, 0)
    res = np.roll(res, -kw//2, 1)


if __name__ == "__main__":

    args = parser.parse_args()

    # parse arguments
    ftype = args.type if args.type is not None else "linear"
    flen = args.length if args.length is not None else 10
    fori = args.angle if args.angle is not None else 0

    # load the image
    img = cv2.imread(args.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if ftype == "linear":
        res = apply_filter(img, ftype, flen, ori=fori)
    elif ftype == "circular":
        res = apply_filter(img, ftype, flen)
    else:
        assert False, "kernel type must be linear or circular"

    # output
    out = cv2.cvtColor(res, cv2.COLOR_GRAY2RGB)
    cv2.imshow("Deblurred Image", out)
    cv2.waitKey()
    cv2.destroyAllWindows()

