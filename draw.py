import cv2
import numpy as np
from matplotlib import pyplot as plt

def draw_matches(src_pts, dst_pts, img1, img2):
	'''Places the 2 images side-by-side and draws lines between matching keypoints.'''
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

def draw_epilines(src_pts, dst_pts, img1, img2):
	# find the fundamental matrix
	F, mask = cv2.findFundamentalMat(src_pts, dst_pts, cv2.RANSAC)

	# select only inlier points
	img1_pts = src_pts[mask.ravel()==1]
	img2_pts = dst_pts[mask.ravel()==1]

	# find epilines corresponding to points in the 2nd image and draw them on the 1st image
	lines1 = cv2.computeCorrespondEpilines(img2_pts.reshape(-1,1,2), 2, F)
	lines1 = lines1.reshape(-1,3)
	img5, img6 = draw_lines(img1, img2, lines1, img1_pts, img2_pts)

	# find epilines corresponding to points in the 1st image and draw them on the 2nd image
	lines2 = cv2.computeCorrespondEpilines(img1_pts.reshape(-1,1,2), 1, F)
	lines2 = lines2.reshape(-1,3)
	img3, img4 = draw_lines(img2, img1, lines2, img2_pts, img1_pts)

	plt.subplot(121), plt.imshow(img5)
	plt.subplot(122), plt.imshow(img3)
	plt.show()

def draw_lines(img1, img2, lines, img1_pts, img2_pts):
	h, w = img1.shape
	for r, pt1, pt2 in zip(lines, img1_pts, img2_pts):
		color = tuple(np.random.randint(0,255,3).tolist())
		x0, y0 = [ int(x) for x in [ 0, -r[2]/r[1] ] ]
		x1, y1 = [ int(x) for x in [ w, -(r[2]+r[0]*w) / r[1] ] ]
		cv2.line(img1, (x0,y0), (x1,y1), color, 1)
		cv2.circle(img1, tuple(pt1[0]), 5, color, -1)
		cv2.circle(img2, tuple(pt2[0]), 5, color, -1)

	return img1, img2