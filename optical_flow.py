import cv2
import numpy as np
import math

class Point3D(object):
    def __init__(self, coords, origin):
        self.coords = coords
        self.origin = origin

def downsample_images(img1, img2):
    '''Downsamples an image pair.'''
    # img1 and img2 are HxWx3 arrays (rows, columns, 3-colour channels)
    img1 = cv2.pyrDown(img1)
    img2 = cv2.pyrDown(img2)

    return img1, img2

def gray_downsampled(img1, img2):
    '''Convert downsampled images to grayscale if the images are found.'''
    try:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    except:
        print "Image not found!"

    return img1_gray, img2_gray

def magnitude(vector):
    return math.sqrt(vector[0]**2 + vector[1]**2)

def calc_flow(img1_gray, img2_gray):
    '''The flow is a HxWx2 array of motion flow vectors.'''
    PYR_SCALE, LEVELS = 0.5, 3
    WINSIZE, ITERATIONS = 15, 10
    POLY_N, POLY_SIGMA = 5, 1.1
    FLAGS = 1

    flow = cv2.calcOpticalFlowFarneback(img1_gray, img2_gray, PYR_SCALE, LEVELS, WINSIZE, ITERATIONS, POLY_N, POLY_SIGMA, FLAGS)
    return flow

def match_points(img2, flow):
    '''Finds points in the second image that matches the first, based on the motion flow vectors.'''
    # min and max magnitudes of the motion flow vector to be included in the reconstruction
    MIN_MAG, MAX_MAG = 1, 100
    # create an empty HxW array to store the dst points
    h, w = img2.shape[0], img2.shape[1]

    src_pts = [ [[col, row]] for row in xrange(h) for col in xrange(w) if (0 < int(row + flow[row, col][0]) < h) and (0 < int(col + flow[row, col][1]) < w) and MIN_MAG < magnitude(flow[row, col]) < MAX_MAG ]
    dst_pts = [ [[int(col + flow[row, col][1]), int(row + flow[row, col][0])]] for row in xrange(h) for col in xrange(w) if (0 < int(row + flow[row, col][0]) < h) and (0 < int(col + flow[row, col][1]) < w) and MIN_MAG < magnitude(flow[row, col]) < MAX_MAG ]
    src_pts = np.array(src_pts)
    dst_pts = np.array(dst_pts)
    # src and dst pts are Nx1x2 arrays that contain the x and y coordinates of the matching points
    return src_pts, dst_pts

def attach_tracks(i, pts_3D, norm_pts1, norm_pts2, pt_cloud_indexed=[]):
    # convert norm_pts to Nx2 arrays
    norm_pts1 = np.array([ pt[0] for pt in norm_pts1 ])
    norm_pts2 = np.array([ pt[0] for pt in norm_pts2 ])

    def find_point(new_pt, pt_cloud_indexed):
        for old_pt in pt_cloud_indexed:
            try:
                if new_pt.origin[i] == old_pt.origin[i]:
                    return True, old_pt
            except KeyError:
                continue
        return False, None

    new_pts = [ Point3D(pt, {i: norm_pts1[num], i+1: norm_pts2[num]}) for num, pt in enumerate(pts_3D) ]

    if pt_cloud_indexed == []:
        pt_cloud_indexed = new_pts
    else:
        for num, new_pt in enumerate(new_pts):
            found, old_pt = find_point(new_pt, pt_cloud_indexed)
            if found:
                old_pt.origin[i+1] = norm_pts2[num]
            else:
                pt_cloud_indexed.append(new_pt)

    return pt_cloud_indexed

def scan_tracks(i, norm_pts1, norm_pts2, pt_cloud_indexed):
    matched_pts_2D = [ norm_pts2[num] for (num, pt_2D) in enumerate(norm_pts1) for pt_3D in pt_cloud_indexed if pt_3D.origin[i][0] == pt_2D[0][0] and pt_3D.origin[i][1] == pt_2D[0][1] ]
    matched_pts_3D = [ pt_3D.coords for (num, pt_2D) in enumerate(norm_pts1) for pt_3D in pt_cloud_indexed if pt_3D.origin[i][0] == pt_2D[0][0] and pt_3D.origin[i][1] == pt_2D[0][1] ]

    matched_pts_2D = np.array(matched_pts_2D, dtype='float32')
    matched_pts_3D = np.array(matched_pts_3D, dtype='float32')

    return matched_pts_2D, matched_pts_3D

def draw_flow(img, flow, step=16):
    # plot optical flow at sample points spaced step pixels apart
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    fx, fy = flow[y,x].T

    # create line endpoints
    lines = np.vstack([x, y, x+fx, y+fy]).T.reshape(-1,2,2)
    lines = np.int32(lines)

    # create image and draw
    vis = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x1, y1), (x2, y2) in lines:
        cv2.line(vis, (x1,y1), (x2,y2), (0,255,0), 1)
        cv2.circle(vis, (x1,y1), 1, (0,255,0), -1)
    return vis


# cv2.imshow('Optical flow', draw_flow(gray, flow))
# if cv2.waitKey(30) == 27:
#     break