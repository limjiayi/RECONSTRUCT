import cv2
import numpy as np
from matplotlib import pyplot as plt
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
	# Initiate detector
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
	kp1, des1, kp2, des2 = find_keypoints_descriptors(img1, img2)
	src_pts, dst_pts = match_keypoints(kp1, des1, kp2, des2)
	# draw.draw_matches(src_pts, dst_pts, img1, img2)
	draw.draw_epilines(src_pts, dst_pts, img1, img2)

main()