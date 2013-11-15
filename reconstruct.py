import os
import cv2
import numpy as np
from scipy.spatial import Delaunay
from gi.repository import GExiv2
import draw, vtk_cloud

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

	def validate(user_input):
		'''Checks if each character in the string (except periods) are integers.'''
		for i in user_input.translate(None, '.'):
			if not 48 <= ord(i) <= 57:
				return False
		return True

	def get_sensor_sizes(i, prev_sensor, metadata1, metadata2):
		'''Displays camera model based on EXIF data.
		Gets user's input on the width of the camera's sensor.'''
		# focal length in pixels = (image width in pixels) * (focal length in mm) / (CCD width in mm)
		if i == 0:
			print "Camera %s is a %s." % (str(i+1), metadata1['Exif.Image.Model'])
			sensor_1 = raw_input("What is the width (in mm) of camera %s's sensor? > " % str(i+1))
			print "Camera %s is a %s." % (str(i+2), metadata2['Exif.Image.Model'])
			sensor_2 = raw_input("What is the width (in mm) of camera %s's sensor? > " % str(i+2))
		elif i >= 1:
			sensor_1 = str(prev_sensor)
			print "Camera %s is a %s." % (str(i+2), metadata2['Exif.Image.Model'])
			sensor_2 = raw_input("What is the width (in mm) of camera %s's sensor? > " % str(i+2))

		if validate(sensor_1) and validate(sensor_2):
			return float(sensor_1), float(sensor_2)

	metadata1 = GExiv2.Metadata(filename1)
	metadata2 = GExiv2.Metadata(filename2)

	if metadata1.get_supports_exif() and metadata2.get_supports_exif():
		sensor_1, sensor_2 = get_sensor_sizes(i, prev_sensor, metadata1, metadata2)
	else:
		if metadata1.get_supports_exif() == False:
			print "Exif data not available for ", filename1
		if metadata2.get_supports_exif() == False:
			print "Exif data not available for ", filename2
		return None

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

	# src_pts and dst_pts are Nx1x2 arrays
	return src_pts, dst_pts

def normalize_pts(K1, K2, src_pts, dst_pts):
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

def find_essential_matrix(norm_pts1, norm_pts2):
	# convert to Nx2 arrays for findFundamentalMat
	norm_pts1 = np.array([ pt[0] for pt in norm_pts1 ])
	norm_pts2 = np.array([ pt[0] for pt in norm_pts2 ])
	E, mask = cv2.findFundamentalMat(norm_pts1, norm_pts2, cv2.RANSAC)

	return E, mask

def find_projection_matrices(E):
	'''Compute the second camera matrix (assuming the first camera matrix = [I 0]).
	Output is a list of 4 possible camera matrices for P2.'''
	# the first camera matrix is assumed to be the identity matrix
	P1 = np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)

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

def refine_points(norm_pts1, norm_pts2, E, mask):
	'''Refine the coordinates of the corresponding points using the Optimal Triangulation Method.'''
	# select only inlier points
	norm_pts1 = norm_pts1[mask.ravel()==1]
	norm_pts2 = norm_pts2[mask.ravel()==1]

	# convert to 1xNx2 arrays for cv2.correctMatches
	refined_pts1 = np.array([ [pt[0] for pt in norm_pts1 ] ])
	refined_pts2 = np.array([ [pt[0] for pt in norm_pts2 ] ])
	refined_pts1, refined_pts2 = cv2.correctMatches(E, refined_pts1, refined_pts2)

	# outputs are also 1xNx2 arrays
	return refined_pts1, refined_pts2

def get_colours(img1, P1, homog_3D):
	'''Extract RGB data from the original images and store them in new arrays.'''
	# project 3D points back to the image plane
	# camera.project returns a 3xN array of homogeneous points --> convert to Nx3 array 
	camera = draw.Camera(P1)
	img_pts = camera.project(homog_3D).T

	# extract RGB information and store in new arrays with the coordinates
	img_colours = np.array([ img1[ pt[1] ][ pt[0] ] for pt in img_pts ])

	return img_colours

def triangulate_points(P1, P2, refined_pts1, refined_pts2):
	'''Reconstructs 3D points by triangulation using Direct Linear Transformation.'''
	# convert to 2xN arrays
	img1_pts = refined_pts1[0].T
	img2_pts = refined_pts2[0].T

	# pick the P2 matrix with the most scene points in front of the cameras after triangulation
	ind = 0
	maxres = 0
	for i in range(4):
		# print "P", i
		# triangulate inliers and compute depth for each camera
		homog_3D = cv2.triangulatePoints(P1, P2[i], img1_pts, img2_pts)
		# the sign of the depth is the 3rd value of the image point after projecting back to the image
		# i.e. the z-value?
		d1 = np.dot(P1, homog_3D)[2]
		# print "num pts: ", np.dot(P1, homog_3D).shape[1]
		d2 = np.dot(P2[i], homog_3D)[2]
		
		# num_true = 0
		# for item in (d1 > 0) & (d2 < 0):
		# 	if item == True:
		# 		num_true += 1
		# print "pts in front: ", num_true

		if sum(d1 > 0) + sum(d2 < 0) > maxres:
			maxres = sum(d1 > 0) + sum(d2 < 0)
			ind = i
			infront = (d1 > 0) & (d2 < 0)

	# print "selected: P", ind
	# triangulate inliers and keep only points that are in front of both cameras
	# homog_3D is a 4xN array of reconstructed points in homogeneous coordinates
	# pts_3D is a Nx3 array where each N contains an x, y and z-coord
	homog_3D = cv2.triangulatePoints(P1, P2[ind], img1_pts, img2_pts)
	homog_3D = homog_3D[:, infront]
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

def gen_pt_cloud(i, prev_sensor, image1, image2):
	'''Generates a point cloud for every pair of images.'''
	img1, img2 = load_images(image1, image2)
	sensor_i, K1, K2 = build_calibration_matrices(i, prev_sensor, image1, image2)

	img1_gray, img2_gray = gray_images(img1, img2)
	kp1, des1, kp2, des2 = find_keypoints_descriptors(img1_gray, img2_gray)
	src_pts, dst_pts = match_keypoints(kp1, des1, kp2, des2)
	norm_pts1, norm_pts2 = normalize_pts(K1, K2, src_pts, dst_pts)

	E, mask = find_essential_matrix(norm_pts1, norm_pts2)
	P1, P2 = find_projection_matrices(E)

	refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E, mask)
	homog_3D, pts_3D = triangulate_points(P1, P2, refined_pts1, refined_pts2)
	img_colours = get_colours(img1, P1, homog_3D)

	return sensor_i, E, P1, homog_3D, pts_3D, img_colours



def main():
	'''Loop through each pair of images, find point correspondences and generate 3D point cloud.
	Multiply each point in each new point cloud by all the essential matrix inverses up to that point 
	to get the 3D point (as seen by camera 1) and append this point to the overall point cloud.'''
	directory = 'images/ucd_building4_all'
	# directory = 'images/ucd_coffeeshack_all'
	# images = ['images/data/alcatraz1.jpg', 'images/data/alcatraz2.jpg']
	# images = ['images/ucd_building4_all/00000000.jpg', 'images/ucd_building4_all/00000001.jpg', 'images/ucd_building4_all/00000002.jpg', 'images/ucd_building4_all/00000003.jpg']
	# images = ['images/Merton1/001.jpg', 'images/Merton1/002.jpg']
	images = sorted([ str(directory + "/" + img) for img in os.listdir(directory) if img.rpartition('.')[2].lower() in ('jpg', 'png', 'pgm', 'ppm') ])
	E_matrices = []
	proj_matrices = []
	prev_sensor = 0

	for i in range(len(images)-1):
		print "Processing ", images[i].split('/')[2], "and ", images[i+1].split('/')[2]
		prev_sensor, E,  P, homog_3D, pts_3D, img_colours = gen_pt_cloud(i, prev_sensor, images[i], images[i+1])
		E_matrices.append(E)
		proj_matrices.append(P)

		if i == 0:
			# first 2 images
			print pts_3D
			pt_cloud = np.array(pts_3D)
			colours = np.array(img_colours)

		elif i >= 1:
			# j keeps track of the essential matrix to be inversed
			for j in reversed(range(i)):
				print "Multiplying the inverse of E_matrices[%s] to each pt" % j
				pts_3D = np.array([ np.dot(np.linalg.inv(proj_matrices[j][:,:3]), pt.T).T for pt in pts_3D ])
				print pts_3D
				# pts_3D = np.array([ np.dot(pt, np.linalg.inv(E_matrices[j])) for pt in pts_3D ])

			print "pt_cloud shape: ", pt_cloud.shape
			print "pts_3D shape: ", pts_3D.shape
			pt_cloud = np.vstack((pt_cloud, pts_3D))
			colours = np.vstack((colours, img_colours))

			print "pt_cloud: ", pt_cloud.shape
			print "colours: ", colours.shape
		 
	# homog_pt_cloud = np.vstack((pt_cloud.T, np.ones(pt_cloud.shape[0])))
	# faces = delaunay(pts_3D)

	# draw.draw_matches(src_pts, dst_pts, img1_gray, img2_gray)
	# draw.draw_epilines(src_pts, dst_pts, img1_gray, img2_gray, F, mask)
	# draw.draw_projected_points(homog_pt_cloud, P)
	np.savetxt('ucd_building4_all.txt', [pt for pt in pt_cloud])
	# pt_cloud = np.loadtxt('ucd_building4_all.txt')
	# draw.display_pyglet(pt_cloud, colours)
	vtk_cloud.vtk_show_points(pt_cloud)


main()