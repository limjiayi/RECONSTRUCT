import cv2
import numpy as np
from scipy.spatial import Delaunay
import draw

def load_images(filename1, filename2):
	'''Loads 2 images.'''
	# img1 and img2 are HxWx3 arrays (rows, columns, 3-colour channels)
	img1 = cv2.imread(filename1)
	img2 = cv2.imread(filename2)

	return img1, img2

def gray_images(img1, img2):
	'''Convert images to grayscale if the images are found.'''
	try:
		img1_gray = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
		img2_gray = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
	except:
		print "Image not found!"

	return img1_gray, img2_gray

def find_keypoints_descriptors(img1, img2):
	'''Detects keypoints and computes their descriptors.'''
	# initiate detector
	detector = cv2.SURF()

	# find the keypoints and descriptors
	kp1, des1 = detector.detectAndCompute(img1,None)
	kp2, des2 = detector.detectAndCompute(img2,None)

	return kp1, des1, kp2, des2

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

	# store all the good matches as per Lowe's ratio test.
	good_matches = []
	for m, n in matches:
	    if m.distance < 0.7 * n.distance:
	        good_matches.append(m)

	if len(good_matches) > MIN_MATCH_COUNT:
	    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
	    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
	else:
	    print "Not enough matches were found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT)

	# src_pts and dst_pts are Nx1x2 arrays
	return src_pts, dst_pts

def find_fundamental_matrix(src_pts, dst_pts):
	F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC)

	return F, mask

def find_projection_matrices(F):
	'''Compute the second camera matrix (assuming the first camera matrix = [I 0]).'''
	# compute the right epipole from the fundamental matrix
	U1, S1, V1 = cv2.SVDecomp(F)
	right_epipole = V1[-1] / V1[-1, 2]

	# compute the left epipole from the transpose of the fundamental matrix
	U2, S2, V2 = cv2.SVDecomp(F.T) 
	left_epipole = V2[-1] / V2[-1, 2]

	# the first camera matrix is assumed to be the identity matrix
	P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)

	# find the 2nd camera matrix
	skew_matrix = np.array([[0, -left_epipole[2], left_epipole[1]], [left_epipole[2], 0, -left_epipole[0]], [-left_epipole[1], left_epipole[0], 0]])
	P2 = np.vstack((np.dot(skew_matrix, F.T).T, left_epipole)).T

	return P1, P2

def get_camera_matrix(projMatrix):
	'''Decompose the projection matrix of a camera to the camera matrix (K), rotation matrix (R) 
	and translation vector t'''
	K, R, t = cv2.decomposeProjectionMatrix(projMatrix)[:3]

	return K, R, t

def refine_points(src_pts, dst_pts, F, mask):
	'''Refine the coordinates of the corresponding points using the Optimal Triangulation Method.'''
	# select only inlier points
	img1_pts = src_pts[mask.ravel()==1]
	img2_pts = dst_pts[mask.ravel()==1]

	# convert to 1xNx2 arrays for cv2.correctMatches
	img1_pts = np.array([ [pt[0] for pt in img1_pts for pt[0] in pt] ])
	img2_pts = np.array([ [pt[0] for pt in img2_pts for pt[0] in pt] ])
	img1_pts, img2_pts = cv2.correctMatches(F, img1_pts, img2_pts)

	# outputs are also 1xNx2 arrays
	return img1_pts, img2_pts

def get_colours(img1, img2, img1_pts, img2_pts):
	'''Extract RGB data from the original images and associate them with the feature points.'''
	# convert to Nx2 arrays of type int
	# assume each N is [x, y]
	img1_pts, img2_pts = img1_pts[0], img2_pts[0]
	img1_pts, img2_pts = img1_pts.astype(int), img2_pts.astype(int)

	# extract RGB information and store in new arrays with the coordinates
	img1_colours = np.array([ img1[ pt[1] ][ pt[0] ] for pt in img1_pts ])
	img2_colours = np.array([ img2[ pt[1] ][ pt[0] ] for pt in img2_pts ])

	return img1_colours, img2_colours

def triangulate_points(P1, P2, img1_pts, img2_pts):
	'''Reconstructs 3D points by triangulation using Direct Linear Transformation.'''
	# convert to 2xN arrays
	img1_pts = img1_pts[0].T
	img2_pts = img2_pts[0].T

	# returns 4xN array of reconstructed points in homogeneous coordinates
	homog_3D = cv2.triangulatePoints(P1, P2, img1_pts, img2_pts)

	# make homogeneous by dividing by the last coordinate
	# pts_3D is a Nx3 array where each N contains an x, y and z-coord
	pts_3D = homog_3D / homog_3D[3]
	pts_3D = pts_3D[:3]
	pts_3D = np.array(list(zip(pts_3D[0], pts_3D[1], pts_3D[2])))

	return homog_3D, pts_3D

def delaunay(pts_3D):
	'''Delaunay triangulation of 3D points.'''
	tri = Delaunay(pts_3D)

	edges = set()

	def add_edge(i, j):
		if (i, j)  in edges or (j, i) in edges:
			# already added
			return
		edges.add( (i,j) )

	for i in xrange(tri.nsimplex): 
	       add_edge(tri.vertices[i,0], tri.vertices[i,1])
	       add_edge(tri.vertices[i,0], tri.vertices[i,2]) 
	       add_edge(tri.vertices[i,0], tri.vertices[i,3])
	       add_edge(tri.vertices[i,1], tri.vertices[i,2]) 
	       add_edge(tri.vertices[i,1], tri.vertices[i,3])
	       add_edge(tri.vertices[i,2], tri.vertices[i,3])


# def project_points(pts_3D, rotationMatrix, translationVector, cameraMatrix):
# 	rotationVector = cv2.Rodrigues(rotationMatrix)[0]
# 	print "3D pts shape: ", pts_3D.shape
# 	print "rvec: ", rotationVector
# 	print "tvec: ", translationVector
# 	print "camera matrix: ", cameraMatrix
# 	img_pts, jacobian = cv2.projectPoints(pts_3D ,rotationVector, translationVector, cameraMatrix, None)

# 	return img_pts

# def perspective_transform(src_pts, dst_pts):
#     '''Finds a perspective transformation between 2 planes and applies it to one image to find
#     the corresponding points in the other image.'''
#     # find M, the perspective transformation of img1 to img2
#     # mask specifies the inlier and outlier points
#     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,2.0)
#     matchesMask = mask.ravel().tolist()

#     h,w = img1.shape
#     src = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
#     # apply perspective transformation to obtain corresponding points
#     dst = cv2.perspectiveTransform(src,M)

def main():
	img1, img2 = load_images('data/alcatraz1.jpg', 'data/alcatraz2.jpg')
	img1_gray, img2_gray = gray_images(img1, img2)
	kp1, des1, kp2, des2 = find_keypoints_descriptors(img1_gray, img2_gray)
	src_pts, dst_pts = match_keypoints(kp1, des1, kp2, des2)

	F, mask = find_fundamental_matrix(src_pts, dst_pts)
	P1, P2 = find_projection_matrices(F)
	K1, R1, t1 = get_camera_matrix(P1)
	K2, R2, t2 = get_camera_matrix(P2)

	img1_pts, img2_pts = refine_points(src_pts, dst_pts, F, mask)
	img1_colours, img2_colours = get_colours(img1, img2, img1_pts, img2_pts)
	homog_3D, pts_3D = triangulate_points(P1, P2, img1_pts, img2_pts)
	# delaunay(pts_3D)

	# draw.draw_matches(src_pts, dst_pts, img1, img2)
	# draw.draw_epilines(src_pts, dst_pts, img1, img2, F, mask)
	draw.draw_projected_points(homog_3D, P2)
	draw.display_pyglet(pts_3D, img1_colours)

main()