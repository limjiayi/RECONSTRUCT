import os, sys
import cv2
import numpy as np
from gi.repository import GExiv2
import cam_db

def sort_images(directory):
    return sorted([ str(directory + "/" + img) for img in os.listdir(directory) if img.rpartition('.')[2].lower() in ('jpg', 'jpeg', 'png', 'pgm', 'ppm') ])

def load_images(filename1, filename2):
    '''Loads 2 images.'''
    # img1 and img2 are HxWx3 arrays (rows, columns, 3-colour channels)
    img1 = cv2.imread(filename1)
    img2 = cv2.imread(filename2)
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    return img1, img2

def build_calibration_matrices(i, prev_sensor, filename1, filename2):
    '''Extract exif metadata from image files, and use them to build the 2 calibration matrices.'''

    def get_sensor_sizes(i, prev_sensor, metadata1, metadata2):
        '''Looks up sensor width from the database based on the camera model.'''
        # focal length in pixels = (image width in pixels) * (focal length in mm) / (CCD width in mm)
        if i == 0:
            sensor_1 = cam_db.get_sensor_size(metadata1['Exif.Image.Model'].upper())
            sensor_2 = cam_db.get_sensor_size(metadata2['Exif.Image.Model'].upper())
        elif i >= 1:
            sensor_1 = prev_sensor
            sensor_2 = cam_db.get_sensor_size(metadata2['Exif.Image.Model'])

        return sensor_1, sensor_2

    metadata1 = GExiv2.Metadata(filename1)
    metadata2 = GExiv2.Metadata(filename2)
    print metadata1['Exif.Image.Model']
    print metadata2['Exif.Image.Model']

    if metadata1.get_supports_exif() and metadata2.get_supports_exif():
            sensor_1, sensor_2 = get_sensor_sizes(i, prev_sensor, metadata1, metadata2)
    else:
        if metadata1.get_supports_exif() == False:
                print "Exif data not available for ", filename1
        if metadata2.get_supports_exif() == False:
                print "Exif data not available for ", filename2
        sys.exit("Please try again.")

    # Calibration matrix for camera 1 (K1)
    f1_mm = metadata1.get_focal_length()
    w1 = metadata1.get_pixel_width()
    h1 = metadata1.get_pixel_height()
    f1_px = (w1 * f1_mm) / sensor_1
    K1 = np.array([[f1_px, 0, w1/2], [0, f1_px, h1/2], [0,0,1]])

    # Calibration matrix for camera 2 (K2)
    f2_mm = metadata2.get_focal_length()
    w2 = metadata2.get_pixel_width()
    h2 = metadata2.get_pixel_height()
    f2_px = (w2 * f2_mm) / sensor_2
    K2 = np.array([[f2_px, 0, w2/2], [0, f2_px, h2/2], [0,0,1]])

    return sensor_2, K1, K2

def gray_images(img1, img2):
    '''Convert images to grayscale if the images are found.'''
    try:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    except:
        print "Image not found!"

    return img1_gray, img2_gray

def normalize_points(K1, K2, src_pts, dst_pts):
    print src_pts.shape, dst_pts.shape
    '''Normalize points by multiplying them with the inverse of the K matrix.'''
    # convert to 3xN arrays by making the points homogeneous
    src_pts = np.vstack((np.array([ pt[0] for pt in src_pts ]).T, np.ones(src_pts.shape[0])))
    dst_pts = np.vstack((np.array([ pt[0] for pt in dst_pts ]).T, np.ones(dst_pts.shape[0])))
    print src_pts.shape, dst_pts.shape

    # normalize with the calibration matrices
    # norm_pts1 and norm_pts2 are 3xN arrays
    norm_pts1 = np.dot(np.linalg.inv(K1), src_pts)
    norm_pts2 = np.dot(np.linalg.inv(K2), dst_pts)

    # convert back to Nx1x2 arrays
    norm_pts1 = np.array([ [pt] for pt in norm_pts1[:2].T ])
    norm_pts2 = np.array([ [pt] for pt in norm_pts2[:2].T ])

    return norm_pts1, norm_pts2

def find_essential_matrix(K, norm_pts1, norm_pts2):
    '''Estimate an essential matrix that satisfies the epipolar constraint for all the corresponding points.'''
    # convert to Nx2 arrays for findFundamentalMat
    norm_pts1 = np.array([ pt[0] for pt in norm_pts1 ])
    norm_pts2 = np.array([ pt[0] for pt in norm_pts2 ])
    # inliers (1 in mask) are features that satisfy the epipolar constraint
    F, mask = cv2.findFundamentalMat(norm_pts1, norm_pts2, cv2.RANSAC)
    E = np.dot(K.T, np.dot(F, K))

    return E, mask

def find_projection_matrices(E, poses):
    '''Compute the second camera matrix (assuming the first camera matrix = [I 0]).
    Output is a list of 4 possible camera matrices for P2.'''
    # the first camera matrix is assumed to be the identity matrix for the first image,
    # or the pose of the camera for the second and subsequent images
    P1 = poses[-1]
        
    # make sure E is rank 2
    U, S, V = np.linalg.svd(E)
    if np.linalg.det(np.dot(U, V)) < 0:
        V = -V
    E = np.dot(U, np.dot(np.diag([1,1,0]), V))

    # create matrices
    W = np.array([[0,-1,0], [1,0,0], [0,0,1]])

    # return all four solutions
    P2 = [np.vstack( (np.dot(U,np.dot(W,V)).T, U[:,2]) ).T, 
          np.vstack( (np.dot(U,np.dot(W,V)).T, -U[:,2]) ).T,
          np.vstack( (np.dot(U,np.dot(W.T,V)).T, U[:,2]) ).T, 
          np.vstack( (np.dot(U,np.dot(W.T,V)).T, -U[:,2]) ).T]

    return P1, P2

def apply_mask(mask, norm_pts1, norm_pts2):
    '''Keep only those points that satisfy the epipolar constraint.'''
    norm_pts1 = norm_pts1[mask.ravel()==1]
    norm_pts2 = norm_pts2[mask.ravel()==1]
    return norm_pts1, norm_pts2

def refine_points(norm_pts1, norm_pts2, E):
    '''Refine the coordinates of the corresponding points using the Optimal Triangulation Method.'''
    # convert to 1xNx2 arrays for cv2.correctMatches
    refined_pts1 = np.array([ [pt[0] for pt in norm_pts1 ] ])
    refined_pts2 = np.array([ [pt[0] for pt in norm_pts2 ] ])
    refined_pts1, refined_pts2 = cv2.correctMatches(E, refined_pts1, refined_pts2)

    # refined_pts are 1xNx2 arrays
    return refined_pts1, refined_pts2

def triangulate_points(P1, P2, refined_pts1, refined_pts2):
    '''Reconstructs 3D points by triangulation using Direct Linear Transformation.'''
    # convert to 2xN arrays
    refined_pts1 = refined_pts1[0].T
    refined_pts2 = refined_pts2[0].T

    # pick the P2 matrix with the most scene points in front of the cameras after triangulation
    ind = 0
    maxres = 0

    for i in range(4):
        # triangulate inliers and compute depth for each camera
        homog_3D = cv2.triangulatePoints(P1, P2[i], refined_pts1, refined_pts2)
        # the sign of the depth is the 3rd value of the image point after projecting back to the image
        d1 = np.dot(P1, homog_3D)[2]
        d2 = np.dot(P2[i], homog_3D)[2]
        
        if sum(d1>0) + sum(d2<0) > maxres:
            maxres = sum(d1>0) + sum(d2<0)
            ind = i
            infront = (d1 > 0) & (d2 < 0)

    # triangulate inliers and keep only points that are in front of both cameras
    # homog_3D is a 4xN array of reconstructed points in homogeneous coordinates
    # pts_3D is a Nx3 array
    homog_3D = cv2.triangulatePoints(P1, P2[ind], refined_pts1, refined_pts2)
    homog_3D = homog_3D[:, infront]
    homog_3D = homog_3D / homog_3D[3]
    pts_3D = np.array(homog_3D[:3]).T

    return homog_3D, pts_3D, infront

def apply_infront_filter(infront, norm_pts1, norm_pts2):
    '''Keep only those points that are in front of the cameras.'''
    norm_pts1 = norm_pts1[infront.ravel()==1]
    norm_pts2 = norm_pts2[infront.ravel()==1]
    return norm_pts1, norm_pts2

def get_colours(img1, K1, K2, norm_pts1, norm_pts2, homog_3D):
    '''Extract RGB data from the original images and store them in new arrays.'''
    # get the original x and y image coords
    norm_pts1 = np.array([ pt[0] for pt in norm_pts1 ])
    norm_pts1 = np.vstack((norm_pts1.T, np.ones(norm_pts1.shape[0])))
    img1_pts = np.dot(K1, norm_pts1).T
    norm_pts2 = np.array([ pt[0] for pt in norm_pts2 ])
    norm_pts2 = np.vstack((norm_pts2.T, np.ones(norm_pts2.shape[0])))
    img2_pts = np.dot(K2, norm_pts2).T

    # extract RGB information from first image and store in new arrays with the coordinates
    img_colours = np.array([ img1[ pt[1] ][ pt[0] ] for pt in img1_pts ])

    return img1_pts, img2_pts, img_colours

def compute_cam_pose(K1, matched_pts_2D, matched_pts_3D, poses):
    rvec, tvec = cv2.solvePnPRansac(matched_pts_3D, matched_pts_2D, K1, None)[0:2]
    rmat = cv2.Rodrigues(rvec)[0]
    pose = np.hstack((rmat, tvec))
    poses.append(pose)
    return poses