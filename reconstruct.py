import cv2
import numpy as np
from scipy.spatial import Delaunay
import draw

def load_images(filename1, filename2):
	'''Reads 2 images and converts them to grayscale if they are found.'''
	img1 = cv2.imread(filename1)
	img2 = cv2.imread(filename2)

	# convert images to grayscale
	try:
		img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
		img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
	except:
		print "Image not found!"

	return img1, img2

def find_keypoints_descriptors(img1, img2):
	'''Detects keypoints and computes their descriptors.'''
	# initiate detector
	detector = cv2.SIFT()

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

	return img1_pts, img2_pts

def triangulate_points(P1, P2, img1_pts, img2_pts):
	'''Reconstructs 3D points by triangulation using Direct Linear Transformation.'''
	# returns 4xN array of reconstructed points in homogeneous coordinates
	pts_4D = cv2.triangulatePoints(P1, P2, img1_pts, img2_pts)

	# get 3D points by dividing by the last coordinate
	pts_3D = pts_4D / pts_4D[3]
	pts_3D = pts_3D[:3]
	pts_3D = np.array(list(zip(pts_3D[0], pts_3D[1], pts_3D[2])))

	return pts_3D

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


def project_points(pts_3D, rotationMatrix, translationVector, cameraMatrix):
	rotationVector = cv2.Rodrigues(rotationMatrix)[0]
	print "3D pts shape: ", pts_3D.shape
	print "rvec: ", rotationVector
	print "tvec: ", translationVector
	print "camera matrix: ", cameraMatrix
	img_pts, jacobian = cv2.projectPoints(pts_3D ,rotationVector, translationVector, cameraMatrix, None)

	return img_pts

def perspective_transform(src_pts, dst_pts):
    '''Finds a perspective transformation between 2 planes and applies it to one image to find
    the corresponding points in the other image.'''
    # find M, the perspective transformation of img1 to img2
    # mask specifies the inlier and outlier points
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,2.0)
    matchesMask = mask.ravel().tolist()

    h,w = img1.shape
    src = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    
    # apply perspective transformation to obtain corresponding points
    dst = cv2.perspectiveTransform(src,M)

def main():
	img1, img2 = load_images('Model_House/house.000.pgm', 'Model_House/house.001.pgm')
	img1, img2 = load_images('Merton1/001.jpg', 'Merton1/002.jpg')
	kp1, des1, kp2, des2 = find_keypoints_descriptors(img1, img2)
	src_pts, dst_pts = match_keypoints(kp1, des1, kp2, des2)

	print src_pts.shape

	# F, mask = find_fundamental_matrix(src_pts, dst_pts)
	F = draw.compute_fundamental(src_pts, dst_pts)

	# P1, P2 = find_projection_matrices(F)
	# print P2

	# check if get_camera_matrix works
	# P1_text = open('Merton1/2D/001.P', 'r').read().strip().split('\n')
	# P1 = np.array([ row.split() for row in P1_text ], dtype=float)
	# P2_text = open('Merton1/2D/002.P', 'r').read().strip().split('\n')
	# P2 = np.array([ row.split() for row in P2_text ], dtype=float)
	# print P2

	# K2, R2, t2 = get_camera_matrix(P2)
	# print "K2: ", K2
	# print "R2: ", R2
	# print "t2: ", t2
	# camera = draw.Camera(P2)
	# K2, R2, t2 = camera.factor()
	# print "K2: ", K2
	# print "R2: ", R2
	# print "t2: ", t2

	# img1_pts, img2_pts = refine_points(src_pts, dst_pts, F, mask)
	# pts_3D = triangulate_points(P1, P2, img1_pts, img2_pts)

	# check if draw_projected_points works
	# p3d_file = open('Merton1/3D/p3d', 'r')
	# pts_3D = np.array([ line.strip().split() for line in open('Merton1/3D/p3d', 'r')])
	# draw.draw_projected_points(pts_3D)

	# delaunay(pts_3D)

	# proj_img1_pts = project_points(pts_3D, R1, t1, K1)
	# proj_img2_pts = project_points(pts_3D, R2, t2, K2)

	# draw.draw_matches(src_pts, dst_pts, img1, img2)
	# draw.draw_epilines(src_pts, dst_pts, img1, img2)
	# draw.draw_projected_points(pts_3D)

main()