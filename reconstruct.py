import os
import cv2
import numpy as np
from scipy.spatial import Delaunay
from scipy import linalg
import draw, sfm, vtk_cloud

def load_images(filename1, filename2):
	'''Loads 2 images.'''
	# img1 and img2 are HxWx3 arrays (rows, columns, 3-colour channels)
	img1 = cv2.imread(filename1)
	img2 = cv2.imread(filename2)
	img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
	img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

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
	kp1, des1 = detector.detectAndCompute(img1, None)
	kp2, des2 = detector.detectAndCompute(img2, None)

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

	# store all the good matches as per Lowe's ratio test
	good_matches = []
	for m, n in matches:
	    if m.distance < 0.7 * n.distance:
	        good_matches.append(m)

	if len(good_matches) > MIN_MATCH_COUNT:
	    src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
	    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
	else:
	    print "Not enough matches were found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT)

	# good = []
	# src_pts = []
	# dst_pts = []

	# ratio test as per Lowe's paper
	# for i,(m,n) in enumerate(matches):
	#     if m.distance < 0.7*n.distance:
	#         good.append(m)
	#         dst_pts.append([kp2[m.trainIdx].pt])
	#         src_pts.append([kp1[m.queryIdx].pt])

	# src_pts = np.float32(src_pts)
	# dst_pts = np.float32(dst_pts)
	# print "after matching: ", src_pts.shape

	# src_pts and dst_pts are Nx1x2 arrays
	return src_pts, dst_pts

def find_fundamental_matrix(src_pts, dst_pts):
	# convert to Nx2 arrays for findFundamentalMat
	src_pts = np.array([ pt[0] for pt in src_pts ])
	dst_pts = np.array([ pt[0] for pt in dst_pts ])
	F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC)

	return F, mask

def find_projection_matrices(F):
	'''Compute the second camera matrix (assuming the first camera matrix = [I 0]).'''
	# compute the right epipole from the fundamental matrix
	U1, S1, V1 = cv2.SVDecomp(F)
	right_epipole = V1[2] / V1[2, 2]

	# compute the left epipole from the transpose of the fundamental matrix
	U2, S2, V2 = cv2.SVDecomp(F.T) 
	left_epipole = V2[2] / V2[2, 2]

	# the first camera matrix is assumed to be the identity matrix
	P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)

	# find the 2nd camera matrix
	skew_matrix = np.array([[0, -left_epipole[2], left_epipole[1]], [left_epipole[2], 0, -left_epipole[0]], [-left_epipole[1], left_epipole[0], 0]])
	P2 = np.vstack((np.dot(skew_matrix, F.T).T, left_epipole)).T

	return P1, P2

def get_camera_matrix(projMatrix):
	'''Decompose the projection matrix of a camera to the camera matrix (K), rotation matrix (R) 
	and translation vector (t)'''
	K, R, t = cv2.decomposeProjectionMatrix(projMatrix)[:3]

	return K, R, t

def refine_points(src_pts, dst_pts, F, mask):
	'''Refine the coordinates of the corresponding points using the Optimal Triangulation Method.'''
	print src_pts.shape

	# select only inlier points
	img1_pts = src_pts[mask.ravel()==1]
	img2_pts = dst_pts[mask.ravel()==1]

	# convert to 1xNx2 arrays for cv2.correctMatches
	img1_pts = np.array([ [pt[0] for pt in img1_pts ] ])
	img2_pts = np.array([ [pt[0] for pt in img2_pts ] ])
	img1_pts, img2_pts = cv2.correctMatches(F, img1_pts, img2_pts)

	# outputs are also 1xNx2 arrays
	return img1_pts, img2_pts

def get_colours(img1, img2, img1_pts, img2_pts):
	'''Extract RGB data from the original images and store them in new arrays.'''
	# convert to Nx2 arrays of type int
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
	# pts_3D is a Nx3 array where each N contains an x, y and z-coord
	homog_3D = cv2.triangulatePoints(P1, P2, img1_pts, img2_pts)
	pts_3D = homog_3D / homog_3D[3]
	pts_3D = np.array(pts_3D[:3]).T

	return homog_3D, pts_3D

def delaunay(pts_3D):
	'''Delaunay triangulation of 3D points.'''
	tri = Delaunay(pts_3D)
	faces = []
	vertices = tri.vertices
	for i in xrange(tri.nsimplex):
		faces.extend([
			(vertices[i,0], vertices[i,1], vertices[i,2]),
			(vertices[i,1], vertices[i,3], vertices[i,2]),
			(vertices[i,0], vertices[i,3], vertices[i,1]),
			(vertices[i,0], vertices[i,2], vertices[i,3])
		])

	return faces

def main():
	# img1, img2 = load_images('images/Merton1/001.jpg', 'images/Merton1/002.jpg')
	# img1, img2 = load_images('images/Dinosaur/viff.000.ppm', 'images/Dinosaur/viff.001.ppm')
	img1, img2 = load_images('images/data/alcatraz1.jpg', 'images/data/alcatraz2.jpg')
	img1_gray, img2_gray = gray_images(img1, img2)
	kp1, des1, kp2, des2 = find_keypoints_descriptors(img1_gray, img2_gray)
	src_pts, dst_pts = match_keypoints(kp1, des1, kp2, des2)
	F, mask = find_fundamental_matrix(src_pts, dst_pts)

	P1, P2 = find_projection_matrices(F)
	K1, R1, t1 = get_camera_matrix(P1)
	K2, R2, t2 = get_camera_matrix(P2)

	img1_pts, img2_pts = refine_points(src_pts, dst_pts, F, mask)
	homog_3D, pts_3D = triangulate_points(P1, P2, img1_pts, img2_pts)
	img1_colours, img2_colours = get_colours(img1, img2, img1_pts, img2_pts)

	f = open('alcatraz.txt', 'r+')
	np.savetxt('alcatraz.txt', [pt for pt in pts_3D])

	# convert from 1xNx2 to Nx1x2 arrays
	img1_pts = np.array([ [pt] for pt in img1_pts[0] ])
	img2_pts = np.array([ [pt] for pt in img2_pts[0] ])
	draw.draw_matches(img1_pts, img2_pts, img1_gray, img2_gray)

	# delaunay(pts_3D)

	# draw.draw_matches(src_pts, dst_pts, img1_gray, img2_gray)
	# draw.draw_epilines(src_pts, dst_pts, img1_gray, img2_gray, F, mask)
	draw.draw_projected_points(homog_3D, P2)
	draw.display_pyglet(pts_3D, img1_colours)
	vtk_cloud.vtk_show_points(pts_3D)


# def main():
# 	'''Loop through each pair of images, find point correspondences and generate 3D point cloud.
# 	Multiply each point in each new point cloud by the cumulative matrix inverse to get the 3D point
# 	(as seen by camera 1) and append this point to the overall point cloud.'''
# 	directory = 'images/Merton1'
# 	# images = ['images/Dinosaur/viff.000.ppm', 'images/Dinosaur/viff.001.ppm']
# 	# images = ['images/data/alcatraz1.jpg', 'images/data/alcatraz2.jpg']
# 	# images = ['images/Merton1/001.jpg', 'images/Merton1/002.jpg']
# 	images = sorted([ str(directory + "/" + img) for img in os.listdir(directory) if img.rpartition('.')[2] in ('jpg', 'png', 'pgm', 'ppm') ])
# 	f_matrices = []
# 	proj_matrices = []

# 	for i in range(len(images)-1):
# 		F,  P, homog_3D, pts_3D, img_colours = gen_pt_cloud(images[i], images[i+1])
# 		f_matrices.append(F)
# 		proj_matrices.append(P)

# 		if i == 0:
# 			# first 2 images
# 			F_inv = np.eye(3)
# 			pt_cloud = list(pts_3D)
# 			colours = list(img_colours)

# 		elif i >= 1:
# 			# find the cumulative matrix inverse
# 			F_inv = np.dot(linalg.inv(f_matrices[i-1]), F_inv)

# 			for pt in pts_3D:
# 				pt = np.dot(pt, F_inv)
# 				pt_cloud.append(pt)

# 			for colour in img_colours:
# 				colours.append(colour)

# 	pt_cloud = np.array(pt_cloud)

# 	norm_matrix = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1000]])
# 	pt_cloud_norm = np.dot(norm_matrix, pt_cloud.T)
# 	print "x vals: \n", "min: ", min(pt_cloud_norm[0]), "max:", max(pt_cloud_norm[0]), "\n", "mean: ", np.mean(pt_cloud_norm[0]), "std dev: ", np.std(pt_cloud_norm[0]), "\n"
# 	print "y vals: \n", "min: ", min(pt_cloud_norm[1]), "max:", max(pt_cloud_norm[1]), "\n", "mean: ", np.mean(pt_cloud_norm[1]), "std dev: ", np.std(pt_cloud_norm[1]), "\n"
# 	print "z vals: \n", "min: ", min(pt_cloud_norm[2]), "max:", max(pt_cloud_norm[2]), "\n", "mean: ", np.mean(pt_cloud_norm[2]), "std dev: ", np.std(pt_cloud_norm[2]), "\n"
# 	center = np.array([np.mean(pt_cloud_norm[0]), np.mean(pt_cloud_norm[1]), np.mean(pt_cloud_norm[2])])
# 	print "center: ", center

# 	homog_pt_cloud = np.vstack((pt_cloud.T, np.ones(pt_cloud.shape[0])))
# 	colours = np.array(colours)
# 	faces = delaunay(pts_3D)

# 	# draw.draw_matches(src_pts, dst_pts, img1_gray, img2_gray)
# 	# draw.draw_epilines(src_pts, dst_pts, img1_gray, img2_gray, F, mask)
# 	draw.draw_projected_points(homog_pt_cloud, P)
# 	draw.display_pyglet(pt_cloud, colours)
# 	# draw.display_pyglet(pt_cloud_norm.T, colours)
# 	# draw.display_mayavi(pt_cloud, colours)
# 	vtk_cloud.vtk_show_points(pts_3D)

# def gen_pt_cloud(image1, image2):
# 	img1, img2 = load_images(image1, image2)
# 	img1_gray, img2_gray = gray_images(img1, img2)
# 	kp1, des1, kp2, des2 = find_keypoints_descriptors(img1_gray, img2_gray)
# 	src_pts, dst_pts = match_keypoints(kp1, des1, kp2, des2)

# 	F, mask = find_fundamental_matrix(src_pts, dst_pts)
# 	P1, P2 = find_projection_matrices(F)
# 	# K1, R1, t1 = get_camera_matrix(P1)
# 	# K2, R2, t2 = get_camera_matrix(P2)

# 	img1_pts, img2_pts = refine_points(src_pts, dst_pts, F, mask)
# 	homog_3D, pts_3D = triangulate_points(P1, P2, img1_pts, img2_pts)
# 	img1_colours, img2_colours = get_colours(img1, img2, img1_pts, img2_pts)

# 	return F, P1, homog_3D, pts_3D, img1_colours

main()