"""
pano.py: A full-stack panorama stitcher.

author: Frank Liu - frank.zijie@gmail.com
last modified: 03/13/2015

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

TODOs:
  - matching using locally distinctive image features
  - a more efficient cylindrical projection
"""

import argparse
from collections import deque
import os
import sys
from timeit import default_timer

import cv2
import numpy as np

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'PNG', 'JPG', 'JPEG'])

parser = argparse.ArgumentParser(description="A full-stack panorama stitcher.",
                                 usage="pano.py <directory>")
parser.add_argument("-i", "--input", default="test", help="path to directory with images for stitching")
parser.add_argument("-p", "--proj", default="cylindrical", help="projection type (planar or cylindrical)")
parser.add_argument("-v", "--verbose", default=1, help="specify the verbosity of the operation (0 or 1)")

# for focal length computation
np.seterr(all="ignore")

# initialize weights for blending
blend_weights = np.zeros((128, 128), dtype=np.float32)
iter_bw = np.nditer(blend_weights, 
                    flags=['multi_index'], 
                    op_flags=['writeonly'])
center = (blend_weights.shape[0]/2.0, blend_weights.shape[1]/2.0)
for p in iter_bw:
    y = 1 - abs(iter_bw.multi_index[0] - center[0]) / (center[0] + 1)
    x = 1 - abs(iter_bw.multi_index[1] - center[1]) / (center[1] + 1)
    p[...] = x * y


class PanoImage:

    # minimum number of feature matches for image match
    MIN_FEAT_MATCHES = 50

    # minimum focal length (small focals lead to distorted panoramas)
    MIN_FOCAL_LENGTH = 10

    # feature detector
    #detector = cv2.ORB(nfeatures=2048)
    detector = cv2.SIFT(nfeatures=1024, edgeThreshold=40)

    # FLANN matcher
    # OpenCV 2.4.9 does not contain FLANN enums
    # ** hopefully will be changed in OpenCV 3+
    #matcher = cv2.FlannBasedMatcher(dict(algorithm=4, 
    #                                     table_number=6,
    #                                     key_size=12,
    #                                     multi_probe_level=1), {})
    matcher = cv2.FlannBasedMatcher(dict(algorithm = 1, 
                                         trees = 5), {})

    def __init__(self, path):
        self.img = cv2.imread(path)
        self.img_bands = []
        self.feat_matches = {}
        self.n_feat_matches = 0
        self.img_matches = {}
        self.H = None # homography to "root" of connected component

    def computeFeatures(self):
        """
            Computes features for this particular image.
        """

        detector = PanoImage.detector

        # extract keypoints and descriptors
        img_gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        (self.keypts, self.descs) = detector.detectAndCompute(img_gray, None)
        self.descs = self.descs.astype(np.float32)

    def matchFeatures(self, pano_img):
        """
            Acquires matching features with another PanoImage instance.
        """

        matcher = PanoImage.matcher

        # get the best feature matches using 2NN heuristic
        matches = matcher.knnMatch(self.descs, pano_img.descs, k=2)
        best_matches = []
        for (m1, m2) in matches:
            if m1.distance < 0.8*m2.distance:
                best_matches.append(m1)

        # image match heuristic
        if len(best_matches) > PanoImage.MIN_FEAT_MATCHES:
            if pano_img not in self.img_matches:

                # all matches
                src_pts = np.array([self.keypts[m.queryIdx].pt for m in best_matches])
                dst_pts = np.array([pano_img.keypts[m.trainIdx].pt for m in best_matches])

                # get feature correspondences
                H, mask = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, 
                                             ransacReprojThreshold=3.0)
                corresp_idxs = np.where(mask)[0]
                src_corresp = src_pts[corresp_idxs]
                dst_corresp = dst_pts[corresp_idxs]

                # add to feature correspondences for bundle adjustment (uni-directional)
                self.feat_matches[pano_img] = (src_corresp, dst_corresp)
                self.n_feat_matches += len(src_corresp)

                # add to image matches
                self.img_matches[pano_img] = H
                pano_img.img_matches[self] = np.linalg.inv(H)

                # project source correspondences onto destination correspondences
                #all_src_proj = cv2.perspectiveTransform(np.array([self.keypts]), H)
                #for i, sp in enumerate(src_proj):
                #    for j, dp in enumerate(pano_img.keypts):
                #        if dist(spt, dpt) < 8 and dist()

            return True

        return False

    def estimateFocalLength(self):
        """
            Estimates focal length using the global perspective transform.
        """

        H = self.H
        f_min = PanoImage.MIN_FOCAL_LENGTH

        # first estimate
        denom = H[0][0]**2 + H[0][1]**2 - H[1][0]**2 - H[1][1]**2
        f_sq = np.divide(H[1][2]**2 - H[0][2]**2, denom)
        f0 = False if f_sq == np.inf or f_sq < f_min else np.sqrt(f_sq)

        # second estimate
        denom = H[0][0] * H[1][0] + H[0][1] * H[1][1]
        f_sq = np.divide(-H[1][2] * H[0][2], denom)
        f1 = False if f_sq == np.inf or f_sq < f_min else np.sqrt(f_sq)

        # third estimate
        #print((np.dot(H[0,:], H[2,:]), np.dot(H[1,:], H[2,:])))

        # return the geometric mean if both estimates exist
        if not f0 and not f1:
            return False
        elif not f0:
            return 2*f1
        elif not f1:
            return 2*f0
        else:
            return 2*np.sqrt(f0*f1)

    def warpMinMax(self, warp_type="planar"):
        """
            Returns the max and min coordinates of the planar warped image.
        """

        (y, x) = self.img.shape[:2]

        # four corners of image
        corners = np.array([[0, 0],
                            [x, 0],
                            [0, y],
                            [x, y]], dtype=np.float32)

        # transform points
        t_corners = cv2.perspectiveTransform(np.array([corners]), self.H).squeeze()
        max_min_coords = np.array((t_corners[:,0].min(), # min X
                                   t_corners[:,1].min(), # min Y
                                   t_corners[:,0].max(), # max X
                                   t_corners[:,1].max()), dtype=np.float32)

        return max_min_coords

    def warpImage(self, pano_dims, warp_type="planar"):
        """
            Performs a transform on the image.
        """

        global blend_weights

        img = self.img.astype(np.float32)
        mask = cv2.resize(blend_weights, (self.img.shape[:2])[::-1])

        # use warpPerspective() for planar warps
        if warp_type == "planar":
            img_warped = cv2.warpPerspective(img, self.H, pano_dims,
                                             borderMode=cv2.BORDER_REPLICATE)
            mask_warped = cv2.warpPerspective(mask, self.H, pano_dims,
                                              flags=cv2.INTER_NEAREST)
        
        #@TODO: cylindrical warp
        elif warp_type == "cylindrical":
            pass

        mask_warped = mask_warped.reshape(mask_warped.shape + (-1,))

        return (img_warped, mask_warped)

def loadImages(paths):

    print("Loading and computing features for images..."),

    # load all images and compute features (ORB)
    pano_imgs = []
    for i, path in enumerate(paths):
        print("{0}".format(i)),
        pano_imgs.append(PanoImage(path))
        pano_imgs[i].computeFeatures()
        print("\b"*(2+len(str(i)))),

    print("done.")

    return pano_imgs

def findImageMatches(pano_imgs):
    """
        Finds image matches using point correspondences.
    """

    print("Finding bidirectional image matches..."),

    # match features between images
    n_matches = 0
    n_imgs = len(pano_imgs)
    for i in range(0, n_imgs):
        for j in range(i+1, n_imgs):
            if pano_imgs[i].matchFeatures(pano_imgs[j]):
                n_matches += 1

    print("found {0} match(es).".format(n_matches))

def findConnectedComponents(pano_imgs):
    """
        Finds connected components of images.
    """

    print("Finding connected components of images..."),

    # shallow copy of all input images
    pimgs = list(pano_imgs)

    # find connected components by image
    conn_comps = [[pimgs.pop()]]
    while len(pimgs) != 0:
        pimg = pimgs.pop()

        # loop through all conn. comps. and image matches
        cc_matches = []
        for cc in conn_comps:
            for im in pimg.img_matches.keys():
                if im in cc: # matching img exists in component
                    if cc not in cc_matches:
                        cc_matches.append(cc)

        # merge all component matches by new image
        if len(cc_matches) > 0:
            cc_matches[0].append(pimg)
            for cc in cc_matches[1:]:
                cc_matches[0].extend(cc)
                conn_comps.remove(cc)
        else:
            conn_comps.append([pimg])

    print("found {0} component(s).".format(len(conn_comps)))

    return conn_comps

def compInitialHomographies(conn_comps):
    """
        Computes initial perspective transforms.
    """

    print("Computing initial perspective transforms..."),

    for i, pimgs in enumerate(conn_comps):

        # Djikstra params
        root = pimgs[0]
        found = [root]
        paths = [[root]]
        new_paths = deque([[root]])

        # continue until connected paths have been found
        while len(found) != len(pimgs):
            p = new_paths.popleft()
            for im in p[-1].img_matches.keys():
                if im not in found:
                    im_path = list(p)+[im]
                    found.append(im)
                    paths.append(im_path)
                    new_paths.append(im_path)

        # compute homographies
        base = paths.pop(0)[0]
        base.H = np.identity(3)
        for p in paths:
            H = np.identity(3)
            for i in range(1, len(p)):
                H = H.dot(p[i].img_matches[p[i-1]])
            p[i].H = H

    print("done.")

def _p_ij(u_i_all, H_i, H_j):
    """Computes the projection for a set of points."""

    # project the point into the image plane
    H = np.linalg.inv(H_j).dot(H_i)
    p_ij = cv2.perspectiveTransform(np.array([u_i_all]), H).squeeze()

    return p_ij

def _dpdp(p_ij_all):
    """Computes dp/dp~ for a collection of points."""

    dpdp_all = []
    for p in p_ij_all:
        dpdp = np.array([[1, 0, -p[0]], [0, 1, -p[1]]])
        dpdp_all.append(dpdp)

    return dpdp_all

def _dpdh_i(u_i_all, H_i, H_j):
    """Computes dp/dh w.r.t H_i."""

    H_j_inv = np.linalg.inv(H_j)
    dpdh_i_all = [np.zeros((3, 8)) for n in range(len(u_i_all))]

    # all 3D versions of input points
    u_3d_all = []
    for l, u in enumerate(u_i_all):
        u_3d = np.hstack((u, 1)).reshape(3, 1)
        u_3d_all.append(u_3d)

    # partials w.r.t. H_i
    for k in range(8):
        dHdh_i = np.zeros((3, 3))
        dHdh_i[k/3,k%3] = 1
        dHdh_ij = H_j_inv.dot(dHdh_i)

        for l, u in enumerate(u_i_all):
            u_3d = u_3d_all[l]
            dpdh_ij = dHdh_ij.dot(u_3d).squeeze()
            dpdh_i_all[l][:,k] = dpdh_ij

    return dpdh_i_all

def _dpdh_j(u_i_all, H_i, H_j):
    """Computes dp/dh w.r.t H_j."""

    H_j_inv = np.linalg.inv(H_j)
    dpdh_j_all = [np.zeros((3, 8)) for n in range(len(u_i_all))]

    u_3d_all = []
    for l, u in enumerate(u_i_all):
        u_3d = np.hstack((u, 1)).reshape(3, 1)
        u_3d_all.append(u_3d)

    # partials w.r.t. H_j
    for k in range(8):
        dHdh_j = np.zeros((3, 3))
        dHdh_j[k/3,k%3] = 1
        dHdh_ij = -H_j_inv.dot(dHdh_j)
        dHdh_ij = dHdh_ij.dot(H_j_inv)
        dHdh_ij = dHdh_ij.dot(H_i)

        for l, u in enumerate(u_i_all):
            u_3d = u_3d_all[l]
            dpdh_ij = dHdh_ij.dot(u_3d).squeeze()
            dpdh_j_all[l][:,k] = dpdh_ij

    return dpdh_j_all

def bundleAdjust(conn_comps):
    """
        Performs bundle adjustment on all connected components.
    """

    print("Performing bundle adjustment..."),

    # perform LMA updates for each connected component
    for i, cc in enumerate(conn_comps):

        # instantiate Jacobian + residual matrices
        n_matches = 0
        for pimg in cc:
            n_matches += pimg.n_feat_matches
        J = np.zeros((n_matches*2, len(cc)*8))
        r = np.zeros((n_matches*2))

        # assign each image a number (cc[0] = root)
        img_nums = {}
        for i, pimg in enumerate(cc):
            img_nums[pimg] = i

        # LM update by error
        delta = 0.001*n_matches
        err = np.inf
        while err == np.inf or err_prev > err + delta:
        #for wtf in range(5):

            # fill Jacobian and residual matrices
            row_n = 0
            for pimg in cc:
                ii = img_nums[pimg]
                for im in pimg.feat_matches:
                    ij = img_nums[im]

                    # load feature matches and compute BA params
                    fm = pimg.feat_matches[im]
                    p_ij = _p_ij(fm[0], pimg.H, im.H)
                    dpdp = _dpdp(p_ij)
                    dpdh_i = _dpdh_i(fm[0], pimg.H, im.H)
                    dpdh_j = _dpdh_j(fm[0], pimg.H, im.H)

                    # fill Jacobian and residual columns
                    for n in range(len(fm[0])):
                        r[row_n:row_n+2] = fm[1][n] - p_ij[n]
                        J[row_n:row_n+2, ii*8:ii*8+8] = dpdp[n].dot(dpdh_i[n])
                        J[row_n:row_n+2, ij*8:ij*8+8] = dpdp[n].dot(dpdh_j[n])
                        row_n += 2

            # compute update vector
            JtJ = np.dot(J.T, J)
            Jtr = np.dot(J.T, r)
            JtJ = JtJ + 0.01*np.eye(len(JtJ))
            update = np.dot(np.linalg.inv(JtJ), Jtr)
            update.resize(update.size/8, 8)

            # compute the error (sigma = 2)
            r_xy = r.reshape(r.size/2, 2)
            errs = r_xy[:,0]**2+r_xy[:,1]**2
            errs_mask = np.where(errs > 2)
            errs[errs_mask] = 4*np.sqrt(errs[errs_mask]) - 4

            # ensure error improvement
            err_prev = err
            err = (np.abs(errs)**2).sum()

            # update each homography
            for i, pimg in enumerate(cc):
                H_up = np.hstack((update[i,:], 0)).reshape(3, 3)
                pimg.H += H_up

    print("done.")

def _blendImagesLinear(pimgs, pano_dims):
    """
        Performs linear blending.
    """

    # instantiate a new panorama and associate a weight image
    pano_shape = pano_dims[::-1]
    pano = np.zeros(pano_shape + (3,), dtype=np.float32)
    weights = np.zeros(pano_shape + (1,), dtype=np.float32)

    # warp the image and add to pano
    for pimg in pimgs:
        (iw, mw) = pimg.warpImage(pano_dims)
        pano += iw * mw
        weights += mw
    
    # weigh each pixel in the panorama
    weights[np.where(weights == 0)] = 1
    pano /= weights

    return pano

def _blendImagesPyramid(pimgs, pano_dims, n_bands=3, sigma=9):
    """
        Performs multi-band (pyramid) blending.
    """

    n_imgs = len(pimgs)

    # create a new panorama
    pano_shape = pano_dims[::-1]
    pano = np.zeros(pano_shape + (3,), dtype=np.float32)
    pano_band = np.zeros(pano_shape + (3,), dtype=np.float32)
    band_weights = np.zeros(pano_shape + (1,), dtype=np.float32)

    # warp weight images and get image bands
    imgs_warped = []
    lin_weights = []
    for pimg in pimgs:
        (iw, pw) = pimg.warpImage(pano_dims)
        imgs_warped.append(iw)
        lin_weights.append(pw)

    # compute mask indexes
    zero_mask = np.zeros(pano_shape + (1,), dtype=np.float32)
    mask_idxs = np.dstack([zero_mask] + lin_weights).argmax(axis=2)
    mask_idxs -= 1

    # generate individual masks
    mask_pano = np.zeros(pano_shape, dtype=np.bool)
    masks_warped = []
    for i in range(n_imgs):
        mask = mask_idxs == i
        mask_pano |= mask
        masks_warped.append(mask.astype(np.float32))
        
    # reconstruct using bands for all images
    reshape_vals = pano_shape + (-1,)
    for n in range(n_bands):
        pano_band[:] = 0
        band_weights[:] = 0
        for i in range(n_imgs):

            # final band
            if n == n_bands - 1:
                img_band = imgs_warped[i]
                mask_band = lin_weights[i]
                #mask_band = masks_warped[i]

            # non-final band (Laplacian pyramid)
            else:
                kernel_size = (sigma*(2*n+1),) * 2
                iw_new = cv2.GaussianBlur(imgs_warped[i], kernel_size, 0)
                masks_warped[i] = cv2.GaussianBlur(masks_warped[i], kernel_size, 0)
                masks_warped[i] = masks_warped[i].reshape(reshape_vals)

                # add contributions to band from individual images
                img_band = imgs_warped[i] - iw_new
                mask_band = masks_warped[i]
                imgs_warped[i] = iw_new

            # add contribution from individual images to band pano
            pano_band += img_band*mask_band
            band_weights += mask_band

        # add contribution to panorama from band pano
        band_weights[np.where(band_weights == 0)] = 1
        pano_band /= band_weights
        pano += pano_band

    # crop the panorama
    pano *= mask_pano.reshape(reshape_vals)

    return (pano, mask_pano.astype(np.uint8))

def _projectOntoCylinder(img, center, focal):
    """
        Performs a cylindrical projection of a planar image.
    """

    if not focal:
        focal = 750

    # define mapping functions
    scale = focal
    mapX = lambda y, x: focal * np.tan(x/scale)
    mapY = lambda y, x: focal / np.cos(x/scale) * y/scale
    def makeMap(y, x):
        map_x = mapX(y - center[1], x - center[0]) + center[0]
        map_y = mapY(y - center[1], x - center[0]) + center[1]
        return np.dstack((map_x, map_y)).astype(np.int16)
    
    # create the LUTs for x and y coordinates
    map_xy = np.fromfunction(makeMap, img.shape[:2], dtype=np.int16)
    img_mapped = cv2.remap(img, map_xy, None, cv2.INTER_NEAREST)

    return img_mapped

def registerPanoramas(conn_comps, projection, blending="pyramid", show_panos=False):
    """
        Registers and displays the panoramas.
    """

    print("Registering panorama(s) using {0} blending...".format(blending)),

    panos = []
    for i, pimgs in enumerate(conn_comps):

        # use the first image in component as "center"
        #@TODO: automatic selection of center image?
        img_anchor = pimgs[0]
        anchor_H = np.linalg.inv(pimgs[0].H)
        for pimg in pimgs:
            pimg.H = pimg.H.dot(anchor_H)

        # estimate the focal length for the panorama
        focal = 1
        count = 0
        for pimg in pimgs:
            f = pimg.estimateFocalLength()
            focal *= f if f else 1
            count += 1 if f else 0
        focal = False if count == 0 else pow(focal, 1.0/count)

        # get the min+max coordinates of each image
        mm_coords = []
        for pimg in pimgs:
            coords = pimg.warpMinMax()
            mm_coords.append(coords)
        mm_coords = np.array(mm_coords)

        # get max+min panorama coordinates
        pano_min_vals = mm_coords[:,:2].min(axis=0)
        pano_max_vals = mm_coords[:,2:].max(axis=0)
        pano_min_max_vals = np.hstack((pano_min_vals, pano_max_vals))

        # get output panorama dimensions and min point (to scale transform)
        n_cols = int(pano_min_max_vals[2] - pano_min_max_vals[0])
        n_rows = int(pano_min_max_vals[3] - pano_min_max_vals[1])
        pano_dims = (n_cols if n_cols % 2 == 0 else n_cols + 1,
                     n_rows if n_rows % 2 == 0 else n_rows + 1)

        # scale all homographies in the component - update by min (x,y)
        move_H = np.eye(3)
        move_H[0,2] -= pano_min_max_vals[0]
        move_H[1,2] -= pano_min_max_vals[1]
        for pimg in pimgs:
            pimg.H = move_H.dot(pimg.H)

        # linear blending
        if blending == "linear":
            (pano, mask) = _blendImagesLinear(pimgs, pano_dims)

        # multi-band (pyramid) blending
        elif blending == "pyramid":
            (pano, mask) = _blendImagesPyramid(pimgs, pano_dims)

        # unknown blending
        else:
            assert False, "{0} blending not implemented".format(blending)

        if projection == "cylindrical":

            # perform a more aesthetic projection
            center_x = -pano_min_max_vals[0] + img_anchor.img.shape[1]/2
            center_y = -pano_min_max_vals[1] + img_anchor.img.shape[0]/2
            pano = _projectOntoCylinder(pano, (center_x, center_y), focal)
            mask = _projectOntoCylinder(mask, (center_x, center_y), focal)

            # remove the black border from the panorama
            cols = np.where(mask.max(axis=0) == 0)
            rows = np.where(mask.max(axis=1) == 0)
            pano = np.delete(pano, rows, axis=0)
            pano = np.delete(pano, cols, axis=1)

        # add panorama to output list
        panos.append(pano)

    print("done.")

    # show panoramas if flag is set
    if show_panos:
        for p in panos:

            # resize maximum dimension to 1000 pixels
            dims = p.shape
            f = 1000.0/max(dims)
            new_dims = (int(dims[1]*f), int(dims[0]*f))
            p_small = cv2.resize(p, new_dims)

            # show the panorama
            win_name = "Panorama {0}".format(i+1)
            cv2.imshow(win_name, p_small)
            cv2.waitKey()
            cv2.destroyWindow(win_name)

    return panos

def build_panoramas(paths, projection="cylindrical", verbose=True):
    """
        Stitches all of the images in a directory.

        Note: this function IS NOT thread-safe (esp. if verbosity is on).
    """

    #@TODO: this is probably not a good idea
    stdout_orig = sys.stdout
    nullstream = open(os.devnull, 'w')
    sys.stdout = sys.stderr if verbose else nullstream

    start = default_timer()

    # stitching pipeline
    # 1) load all images and extract features
    # 2) find image matches using RANSAC
    # 3) find connected components of images
    # 4) use Dijkstra's algorithm to compute initial transforms
    # 5) perform bundle adjustment
    # 6) panorama registration
    pano_imgs = loadImages(paths)
    findImageMatches(pano_imgs)
    conn_comps = findConnectedComponents(pano_imgs)
    compInitialHomographies(conn_comps)
    bundleAdjust(conn_comps)
    panos = registerPanoramas(conn_comps, projection)

    # timing information
    print("Total time elapsed: {0}s".format(default_timer() - start))

    nullstream.close()
    sys.stdout = stdout_orig

    return panos

if __name__ == "__main__":

    args = parser.parse_args()

    # get all images - default to "test_imgs" dir if not specified on cl
    paths = []
    for fname in os.listdir(args.input):
        if fname.split(".")[-1] in ALLOWED_EXTENSIONS:
            paths.append(os.path.join(args.input, fname))
    panos = build_panoramas(paths, projection=args.proj, verbose=args.verbose)

    # save panoramas
    for i, p in enumerate(panos):
        cv2.imwrite("pano{0}.jpg".format(i), p)

