import os, sys
import cv2
import numpy as np
from gi.repository import GExiv2
import pcl
import display_vtk
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

def build_calibration_matrices(i, prev_sensor, K_matrices, filename1, filename2):
    '''Extract exif metadata from image files, and use them to build the 2 calibration matrices.'''

    def get_sensor_sizes(i, prev_sensor, metadata1, metadata2):
        '''Looks up sensor width from the database based on the camera model.'''
        # focal length in pixels = (image width in pixels) * (focal length in mm) / (CCD width in mm)
        if i == 0:
            sensor_1 = cam_db.get_sensor_size(metadata1['Exif.Image.Model'].strip().upper())
            sensor_2 = cam_db.get_sensor_size(metadata2['Exif.Image.Model'].strip().upper())
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

    if i == 0:
        K_matrices.append(K1)
        K_matrices.append(K2)
    elif i >= 1:
        K_matrices.append(K2)

    return sensor_2, K_matrices

def gray_images(img1, img2):
    '''Convert images to grayscale if the images are found.'''
    try:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    except:
        print "Image not found!"

    return img1_gray, img2_gray

def normalize_points(K_matrices, src_pts, dst_pts):
    '''Normalize points by multiplying them with the inverse of the K matrix.'''
    # convert to 3xN arrays by making the points homogeneous
    src_pts = np.vstack((np.array([ pt[0] for pt in src_pts ]).T, np.ones(src_pts.shape[0])))
    dst_pts = np.vstack((np.array([ pt[0] for pt in dst_pts ]).T, np.ones(dst_pts.shape[0])))

    # normalize with the calibration matrices
    # norm_pts1 and norm_pts2 are 3xN arrays
    K1 = K_matrices[-2]
    K2 = K_matrices[-1]
    norm_pts1 = np.dot(np.linalg.inv(K1), src_pts)
    norm_pts2 = np.dot(np.linalg.inv(K2), dst_pts)

    # convert back to Nx1x2 arrays
    norm_pts1 = np.array([ [pt] for pt in norm_pts1[:2].T ])
    norm_pts2 = np.array([ [pt] for pt in norm_pts2[:2].T ])

    return norm_pts1, norm_pts2

def find_essential_matrix(K_matrices, norm_pts1, norm_pts2):
    '''Estimate an essential matrix that satisfies the epipolar constraint for all the corresponding points.'''
    # K = K1, the calibration matrix of the first camera of the current image pair 
    K = K_matrices[-2]
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
        
        if sum(d1 > 0) + sum(d2 < 0) > maxres:
            maxres = sum(d1 > 0) + sum(d2 < 0)
            ind = i
            infront = (d1 > 0) & (d2 < 0)

    # triangulate inliers and keep only points that are in front of both cameras
    # homog_3D is a 4xN array of reconstructed points in homogeneous coordinates, pts_3D is a Nx3 array
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

def filter_outliers(pts_3D, norm_pts1, norm_pts2):
    '''Remove points that are too far away from the median.'''
    x, y, z = pts_3D.T[0], pts_3D.T[1], pts_3D.T[2]
    x_med, y_med, z_med = np.median(x), np.median(y), np.median(z)
    x_std, y_std, z_std = np.std(x), np.std(y), np.std(z)

    N = 2 # number of std devs
    x_mask = [ True if ( x_med - N*x_std < coord < x_med + N*x_std) else False for coord in x ]
    y_mask = [ True if ( y_med - N*y_std < coord < y_med + N*y_std) else False for coord in y ]
    z_mask = [ True if ( z_med - N*z_std < coord < z_med + N*z_std) else False for coord in z ]
    mask = [ all(tup) for tup in zip(x_mask, y_mask, z_mask) ]

    pts_3D = [ pt[0] for pt in zip(pts_3D, mask) if pt[1] ]
    norm_pts1 = [ pt[0] for pt in zip(norm_pts1, mask) if pt[1] ]
    norm_pts2 = [ pt[0] for pt in zip(norm_pts2, mask) if pt[1] ]

    return pts_3D, norm_pts1, norm_pts2

def get_colours(img1, K_matrices, norm_pts1, norm_pts2):
    '''Extract RGB data from the original images and store them in new arrays.'''
    K1 = K_matrices[-2]
    K2 = K_matrices[-1]
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

def compute_cam_pose(K_matrices, matched_pts_2D, matched_pts_3D, poses):
    '''Compute the camera pose from a set of 3D and 2D correspondences.'''
    K1 = K_matrices[-2]
    rvec, tvec = cv2.solvePnPRansac(matched_pts_3D, matched_pts_2D, K1, None)[0:2]
    rmat = cv2.Rodrigues(rvec)[0]
    pose = np.hstack((rmat, tvec))
    poses.append(pose)
    return poses

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

    # display_vtk.vtk_show_points(pt_cloud, list(colours))

def rgb_to_int(colours):
    return np.array([ c[0]*256*256 + c[1]*256 + c[2] for c in colours ])

def save_points(choice, images, pt_cloud, colours, file_path=None, save_format='txt'):
    '''Saves point cloud data in .txt or .pcd formats.'''
    if file_path is None:
        file_path = 'points/' + images[0].split('/')[1].lower() + '_' + choice

    if save_format == 'txt':
        data = np.hstack((pt_cloud, colours))
        np.savetxt('%s.%s' % (file_path, save_format), data, delimiter=" ")

    elif save_format == 'pcd':
        header = ['# .PCD v.7 - Point Cloud Data file format', 
                      'VERSION.7', 'FIELDS x y z rgb', 'SIZE 4 4 4 4', 
                      'TYPE F F F F', 'COUNT 1 1 1 1', 'WIDTH %s' % pt_cloud.shape[0], 
                      'HEIGHT 1', 'VIEWPOINT = 0 0 0 1 0 0 0', 
                      'POINTS %s' % pt_cloud.shape[0], 'DATA ascii']

        colours = rgb_to_int(colours)
        data = np.vstack((pt_cloud.T, colours)).T

        with open('%s.%s' % (file_path, save_format), 'w') as f:
            for item in header:
                f.write(item + '\n')
            for pt in data:
                f.write(np.array_str(pt).strip('[]') + '\n')
    # print "    Saved file as %s.%s" % (file_path.rpartition('/')[2], save_format)

def remove_outliers(file_path):
    points = pcl.PointCloud()
    points.from_file(file_path)

    fil = points.make_statistical_outlier_filter()
    fil.set_mean_k(50)
    fil.set_std_dev_mul_thresh(1.0)
    fil.filter().to_file(file_path.rpartition('.')[0] + '_inliers' + '.pcd')

def check_line(orig_file, line):
    line = [ round(float(i), 5) for i in line.split() ]

    with open(orig_file, 'r') as file_o:
        for j, line_o in enumerate(file_o):
            line_o = [ round(float(k), 5) for k in line_o.split() ]

            if line == line_o[:3]:
                return True, ' '.join( [ str(val) for val in line_o ] )
    return False, None

def pcd_to_txt(orig_file, file_read):
    file_write = file_read.rpartition('.')[0] + '.txt'

    try:
        with open(file_read, 'r') as file_r:
            with open(file_write, 'w') as file_w:
                for i, line in enumerate(file_r):
                    if i > 10:
                        found, line_o = check_line(orig_file, line)
                        if found:
                            file_w.write(line_o + '\n')
    except IOError as e:
        print 'Operation failed: %s' % e.strerror

def extract_points(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def write_points_ba(pt_cloud_indexed, num_views, K_matrices, poses):
    with open('ba.txt', 'w') as f:
        # no. of 3D points, views and 2D measurements
        M = len(pt_cloud_indexed)   # number of 3D points
        N = num_views               # number of views: len(images)
        K = reduce(lambda x, y: x+y, [ len(pt.origin) for pt in pt_cloud_indexed ]) # number of 2D points
        f.write(' '.join([ str(i) for i in [M, N, K] ]) + '\n')

        # K matrices
        K_matrices = [ matrix.ravel().tolist() for matrix in K_matrices ]
        K_matrices = [ [ matrix[0], matrix[1], matrix[2], matrix[4], matrix[5], 0, 0, 0, 0 ] for matrix in K_matrices ]
        for idx, matrix in enumerate(K_matrices):
            f.write(' '.join([str(idx)] + [ str(i) for i in matrix ]) + '\n')

        # 3D point positions
        for idx, pt in enumerate(pt_cloud_indexed):
            coords = pt.coords.tolist()
            f.write(' '.join([ str(idx), str(coords[0]), str(coords[1]), str(coords[2]) ]) + '\n')

        # camera poses
        poses = [ pose.ravel().tolist() for pose in poses ]
        for idx, pose in enumerate(poses):
            f.write(' '.join([ str(i) for i in pose ]) + '\n')

        # 2D image measurements
        for idx, pt in enumerate(pt_cloud_indexed):
            for key in pt.origin:
                f.write(' '.join([ str(key), str(idx), str(pt.origin[key][0][0]), str(pt.origin[key][0][1]), '1' ]) + '\n')

def load_refined(filename):
    with open(filename, 'w') as f:
        pass