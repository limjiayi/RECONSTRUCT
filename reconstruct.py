import os, sys
import cv2
import numpy as np
from scipy.spatial import cKDTree
from gi.repository import GExiv2
import draw, display_vtk, cam_db

def load_points(filename):
    '''Loads .txt and .pcd files.'''
    format = filename.rpartition('.')[2]
    if format == 'txt':
        data = np.loadtxt(filename)
        pt_cloud = data[:,:3]
        colours = data[:,3:]

    elif format == 'pcd':
        data = np.loadtxt(filename)[11:]
        pt_cloud = data[:,:3]
        colours = data[:,3:]

    display_vtk.vtk_show_points(pt_cloud, list(colours))

def save_points(images, pt_cloud, colours, filename=None, save_format='txt'):
    '''Saves point cloud data in .txt or .pcd formats.'''
    if filename is None:
        filename = "points/" + images[0].split('/')[1].lower()

    if save_format == 'txt':
        data = np.hstack((pt_cloud, colours))
        np.savetxt('%s.%s' % (filename, save_format), data, delimiter=" ")

    elif save_format == 'pcd':
        header = list('# .PCD v.7 - Point Cloud Data file format', 
                      ' VERSION.7', 'FIELDS x y z rgb', 'SIZE 4 4 4 4', 
                      'TYPE F F F F', 'COUNT 1 1 1 1', 'WIDTH %s' % pt_cloud.shape[0], 
                      'HEIGHT 1', 'VIEWPOINT = 0 0 0 1 0 0 0', 
                      'POINTS %s' % pt_cloud.shape[0], 'DATA ascii')
        data = np.hstack((pt_cloud, colours))

        f = open('%s.%s' % (filename, save_format), 'w')
        for item in header:
            f.write(item)
        for pt in data:
            f.write(pt)
        f.close()
    print "    Saved file as %s.%s" % (filename, save_format)

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

def find_keypoints_descriptors(img):
    '''Detects keypoints and computes their descriptors.'''
    # initiate detector
    detector = cv2.SURF()

    # find the keypoints and descriptors
    kp, des = detector.detectAndCompute(img, None)

    return kp, des

def match_keypoints(kp1, des1, kp2, des2):
    '''Matches the descriptors in one image with those in the second image using
    the Fast Library for Approximate Nearest Neighbours (FLANN) matcher.'''
    MIN_MATCH_COUNT = 10

    # FLANN parameters
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary

    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(np.asarray(des1, np.float32), np.asarray(des2, np.float32), k=2)

    # store all the good matches as per Lowe's ratio test
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
        # filtered keypoints are lists containing the indices into the keypoints and descriptors
        filtered_kp1 = np.array([ m.queryIdx for m in good_matches ])
        filtered_kp2 = np.array([ m.trainIdx for m in good_matches ])

    else:
        print "Not enough matches were found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT)

    # src_pts and dst_pts are Nx1x2 arrays that contain the x and y coordinates
    return src_pts, dst_pts, filtered_kp1, filtered_kp2

def normalize_pts(K1, K2, src_pts, dst_pts):
    '''Normalize points by multiplying them with the inverse of the K matrix.'''
    # convert to 3xN arrays by making the points homogeneous
    src_pts = np.vstack((np.array([ pt[0] for pt in src_pts ]).T, np.ones(src_pts.shape[0])))
    dst_pts = np.vstack((np.array([ pt[0] for pt in dst_pts ]).T, np.ones(dst_pts.shape[0])))

    # normalize with the calibration matrices
    # norm_pts1 and norm_pts2 are 3xN arrays
    norm_pts1 = np.dot(np.linalg.inv(K1), src_pts)
    norm_pts2 = np.dot(np.linalg.inv(K2), dst_pts)

    # convert back to Nx1x2 arrays
    norm_pts1 = np.array([ [pt] for pt in norm_pts1[:2].T ])
    norm_pts2 = np.array([ [pt] for pt in norm_pts2[:2].T ])

    return norm_pts1, norm_pts2

def find_essential_matrix(K, norm_pts1, norm_pts2):
    # convert to Nx2 arrays for findFundamentalMat
    norm_pts1 = np.array([ pt[0] for pt in norm_pts1 ])
    norm_pts2 = np.array([ pt[0] for pt in norm_pts2 ])
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

def refine_points(norm_pts1, norm_pts2, E, mask, filtered_kp1, filtered_kp2):
    '''Refine the coordinates of the corresponding points using the Optimal Triangulation Method.'''
    # filtered_kp are lists containing the indices into the keypoints and descriptors
    filtered_kp1 = filtered_kp1[mask.ravel()==1]
    filtered_kp2 = filtered_kp2[mask.ravel()==1]

    # convert to 1xNx2 arrays for cv2.correctMatches
    refined_pts1 = np.array([ [pt[0] for pt in norm_pts1 ] ])
    refined_pts2 = np.array([ [pt[0] for pt in norm_pts2 ] ])
    refined_pts1, refined_pts2 = cv2.correctMatches(E, refined_pts1, refined_pts2)

    # refined_pts are 1xNx2 arrays
    return refined_pts1, refined_pts2, filtered_kp1, filtered_kp2

def triangulate_points(P1, P2, refined_pts1, refined_pts2):
    '''Reconstructs 3D points by triangulation using Direct Linear Transformation.'''
    # convert to 2xN arrays
    refined_pts1 = refined_pts1[0].T
    refined_pts2 = refined_pts2[0].T

    # pick the P2 matrix with the most scene points in front of the cameras after triangulation
    ind = 0
    maxres = 0

   # P2 = [ np.dot(np.vstack((P1, np.array([0,0,0,1]))), np.vstack((P, np.array([0,0,0,1]))))[:3] for P in P2 ]
   # print type(P2[0])

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

def filter_norm_pts(mask, infront, norm_pts1, norm_pts2):
    norm_pts1 = norm_pts1[mask.ravel()==1]
    norm_pts2 = norm_pts2[mask.ravel()==1]
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

class Point3D(object):
    def __init__(self, coords, origin):
        self.coords = coords
        self.origin = origin

def attach_indices(i, pts_3D, filtered_kp1, filtered_kp2, pt_cloud_indexed=[]):
    '''Attach to each 3D point, indices into the original lists of keypoints and descriptors 
    of the 2D points that contributed to this 3D point in the cloud.'''

    def find_point(new_pt, pt_cloud_indexed):
        for old_pt in pt_cloud_indexed:
            if new_pt[0] == old_pt.coords[0] and new_pt[1] == old_pt.coords[1] and new_pt[2] == old_pt.coords[2]:
                return True, old_pt
        return False, None

    if pt_cloud_indexed == []:
        for num, pt in enumerate(pts_3D):
            new_pt = Point3D(pt, {i: filtered_kp1[num], i+1: filtered_kp2[num]})
            pt_cloud_indexed.append(new_pt)
    else:
        for num, new_pt in enumerate(pts_3D):
            found, old_pt = find_point(new_pt, pt_cloud_indexed)
            if found:
                old_pt.origin[i] = filtered_kp2[num]
            else:
                new_pt = Point3D(new_pt, {i: filtered_kp1[num], i+1: filtered_kp2[num]})
                pt_cloud_indexed.append(new_pt)

    return pt_cloud_indexed

def gen_pt_cloud(i, prev_sensor, image1, image2, poses):
    '''Generates a point cloud for a pair of images. Generated point cloud is on the local coordinate system.'''
    print "    Loading images..."
    img1, img2 = load_images(image1, image2)
    img1_gray, img2_gray = gray_images(img1, img2)
    print "    Building camera calibration matrices..."
    sensor_i, K1, K2 = build_calibration_matrices(i, prev_sensor, image1, image2)

    print "    Detecting keypoints...\n    Computing descriptors..."
    kp1, des1 = find_keypoints_descriptors(img1_gray)
    kp2, des2 = find_keypoints_descriptors(img2_gray)
    print "    Matching keypoints..."
    src_pts, dst_pts, filtered_kp1, filtered_kp2 = match_keypoints(kp1, des1, kp2, des2)
    norm_pts1, norm_pts2 = normalize_pts(K1, K2, src_pts, dst_pts)

    E, mask = find_essential_matrix(K1, norm_pts1, norm_pts2)
    P1, P2 = find_projection_matrices(E, poses)
    refined_pts1, refined_pts2, filtered_kp1, filtered_kp2 = refine_points(norm_pts1, norm_pts2, E, mask, filtered_kp1, filtered_kp2)
    
    print "    Triangulating 3D points..."
    homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
    norm_pts1, norm_pts2 = filter_norm_pts(mask, infront, norm_pts1, norm_pts2)

    print "    Extracting colour information..."
    img1_pts, img2_pts, img_colours = get_colours(img1, K1, K2, norm_pts1, norm_pts2, homog_3D)
    pt_cloud_indexed = attach_indices(i, pts_3D, filtered_kp1, filtered_kp2)

    return sensor_i, kp2, des2, filtered_kp2, homog_3D, pts_3D, img_colours, pt_cloud_indexed

def scan_cloud(i, prev_kp, prev_des, prev_filter, filtered_kp, pt_cloud_indexed):
    '''Check for matches between the new frame and the current point cloud.'''
    # prev_filter contains the indices into the list of keypoints from the second image in the last iteration
    # filtered_kp contains the indices into the list of keypoints from the first image in the current iteration
    # the second image in the last iteration is the first image in the current iteration
    # therefore, check for matches by comparing the indices
    indices_2D  = []
    matched_pts_2D = []
    matched_pts_3D = []

    for new_idx in filtered_kp:
        for old_idx in prev_filter:
            if new_idx == old_idx:
                # found a match: a keypoint that contributed to both the last and current point clouds
                indices_2D.append(new_idx)

    for idx in indices_2D:
        # pt_cloud_indexed is a list of 3D points from the previous cloud with their keypoint indices
        for pt in pt_cloud_indexed:
            try:
                if pt.origin[i] == idx:
                    matched_pts_3D.append( pt.coords )
                    break
            except KeyError:
                continue
        continue

    for idx in indices_2D:
        matched_pts_2D.append( prev_kp[idx].pt )

    matched_pts_2D = np.array(matched_pts_2D, dtype='float32')
    matched_pts_3D = np.array(matched_pts_3D, dtype='float32')

    return matched_pts_2D, matched_pts_3D

def compute_cam_pose(K1, matched_pts_2D, matched_pts_3D, poses):
    rvec, tvec = cv2.solvePnPRansac(matched_pts_3D, matched_pts_2D, K1, None)[0:2]
    rmat = cv2.Rodrigues(rvec)[0]
    pose = np.hstack((rmat, tvec))
    poses.append(pose)
    return poses

def find_new_pts(i, prev_sensor, image1, image2, prev_kp, prev_des, prev_filter, poses, pt_cloud_indexed):
    print "    Loading images..."
    img1, img2 = load_images(image1, image2)
    img1_gray, img2_gray = gray_images(img1, img2)
    print "    Building camera calibration matrices..."
    sensor_i, K1, K2 = build_calibration_matrices(i, prev_sensor, image1, image2)

    print "    Detecting keypoints...\n    Computing descriptors..."
    new_kp, new_des = find_keypoints_descriptors(img2_gray)
    print "    Matching keypoints..."
    src_pts, dst_pts, filtered_kp1, filtered_kp2 = match_keypoints(prev_kp, prev_des, new_kp, new_des)
    norm_pts1, norm_pts2 = normalize_pts(K1, K2, src_pts, dst_pts)

    E, mask = find_essential_matrix(K1, norm_pts1, norm_pts2)
    refined_pts1, refined_pts2, filtered_kp1, filtered_kp2 = refine_points(norm_pts1, norm_pts2, E, mask, filtered_kp1, filtered_kp2)
    print "    Scanning cloud..."
    matched_pts_2D, matched_pts_3D = scan_cloud(i, prev_kp, prev_des, prev_filter, filtered_kp1, pt_cloud_indexed)
    print "    Computing camera pose..."
    poses = compute_cam_pose(K1, matched_pts_2D, matched_pts_3D, poses)
    P1, P2 = find_projection_matrices(E, poses)

    print "    Triangulating 3D points..."
    homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
    norm_pts1, norm_pts2 = filter_norm_pts(mask, infront, norm_pts1, norm_pts2)

    print "    Extracting colour information..."
    img1_pts, img2_pts, img_colours = get_colours(img1, K1, K2, norm_pts1, norm_pts2, homog_3D)
    updated_pt_cloud_indexed = attach_indices(i, pts_3D, filtered_kp1, filtered_kp2, pt_cloud_indexed)

    return sensor_i, new_kp, new_des, filtered_kp2, poses, homog_3D, pts_3D, img_colours, updated_pt_cloud_indexed


def start(images, filename=None):
    '''Loop through each pair of images, find point correspondences and generate 3D point cloud.
    Multiply each point cloud by the inverse of the camera matrices (camera poses) up to that point 
    to get the 3D point (as seen by camera 1) and append this point cloud to the overall point cloud.'''
    prev_sensor = 0
    poses = [np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)]
    save_format = 'txt'

    for i in range(len(images)-1):
        print images[i]
        print "\n  Processing image %d and %d... " % (i+1, i+2)

        if i == 0:
            # first 2 images
            prev_sensor, prev_kp, prev_des, prev_filter, homog_3D, pts_3D, img_colours, pt_cloud_indexed = gen_pt_cloud(i, prev_sensor, images[i], images[i+1], poses)
            pt_cloud = np.array(pts_3D)
            colours = np.array(img_colours)

        elif i >= 1:
            try:
                prev_sensor, prev_kp, prev_des, prev_filter, poses, homog_3D, pts_3D, img_colours, pt_cloud_indexed = find_new_pts(i, prev_sensor, images[i], images[i+1], prev_kp, prev_des, prev_filter, poses, pt_cloud_indexed)
                pt_cloud = np.vstack((pt_cloud, pts_3D))
                colours = np.vstack((colours, img_colours))
            except:
                print "Error occurred in OpenCV."
                break

    # homog_pt_cloud = np.vstack((pt_cloud.T, np.ones(pt_cloud.shape[0])))
    # draw.draw_matches(src_pts, dst_pts, img1_gray, img2_gray)
    # draw.draw_epilines(src_pts, dst_pts, img1_gray, img2_gray, F, mask)
    # draw.draw_projected_points(homog_pt_cloud, P)

    save_points(images, pt_cloud, colours, filename)
    # display_vtk.vtk_show_points(pt_cloud, list(colours))

def extract_points(filename):
    return np.loadtxt(filename)

def sort_images(directory):
    return sorted([ str(directory + "/" + img) for img in os.listdir(directory) if img.rpartition('.')[2].lower() in ('jpg', 'jpeg', 'png', 'pgm', 'ppm') ])

def main():
    load_filename = ''#points/ucd_building1_all.txt'

    if load_filename != '':
        load_points(load_filename)
    else:
        directory = 'images/ucd_building4_all'
        # images = ['images/data/alcatraz1.jpg', 'images/data/alcatraz2.jpg']
        images = sort_images(directory)
        start(images)


if __name__ == "__main__":
    main()