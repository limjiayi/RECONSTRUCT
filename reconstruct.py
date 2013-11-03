import cv2
import numpy as np
from matplotlib import pyplot as plt

def load_images(filename1, filename2):
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
	# Initiate ORB detector
	orb = cv2.ORB()

	# find the keypoints and descriptors with SIFT
	kp1, des1 = orb.detectAndCompute(img1,None)
	kp2, des2 = orb.detectAndCompute(img2,None)

	return kp1, des1, kp2, des2

def match_keypoints(kp1, des1, kp2, des2, img1):
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

	    # find M, the perspective transformation of img1 to img2
	    # mask specifies the inlier and outlier points
	    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,2.0)
	    matchesMask = mask.ravel().tolist()

	    h,w = img1.shape
	    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	    # apply perspective transformation to obtain corresponding points
	    dst = cv2.perspectiveTransform(pts,M)

	    # img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)

	else:
	    print "Not enough matches were found - %d/%d" % (len(good_matches), MIN_MATCH_COUNT)

	return src_pts, dst_pts

def draw_matches(src_pts, dst_pts, img1, img2):
	# draw_params = dict(matchColor = (0,255,0), # draw matches in green color
 #                    singlePointColor = None,
 #                    matchesMask = matchesMask, # draw only inliers
 #                    flags = 2)

	# img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)

	# plt.imshow(img3, 'gray'),plt.show()

	img1_H, img1_W = img1.shape[:2]
	img2_H, img2_W = img2.shape[:2]

	img_matches = np.zeros((max(img1_H, img2_H), img1_W+img2_W), np.uint8)
	img_matches[:, :img1_W] = img1 # place img1 on the left half
	img_matches[:, img1_W:] = img2 # place img2 on the right half

	for i in range(len(src_pts)):
		pt_a = (int(src_pts[i][0][0]), int(src_pts[i][0][1]))
		pt_b = (int(dst_pts[i][0][0]+img1_W), int(dst_pts[i][0][1]))
		cv2.line(img_matches, pt_a, pt_b, (255,255,255))

	plt.imshow(img_matches,), plt.show()

def main():
	img1, img2 = load_images('Model_House/house.000.pgm', 'Model_House/house.001.pgm')
	kp1, des1, kp2, des2 = find_keypoints_descriptors(img1, img2)
	src_pts, dst_pts = match_keypoints(kp1, des1, kp2, des2, img1)
	draw_matches(src_pts, dst_pts, img1, img2)

main()