import sys
import numpy as np
import draw, display_vtk
from processing import *
from rich_features import *
from optical_flow import *

def gen_pt_cloud(i, prev_sensor, K_matrices, image1, image2, poses, choice):
    '''Generates a point cloud for a pair of images.'''
    print "    Loading images..."
    img1, img2 = load_images(image1, image2)
    img1_gray, img2_gray = gray_images(img1, img2)
    print "    Building camera calibration matrices..."
    sensor_i, K_matrices = build_calibration_matrices(i, prev_sensor, K_matrices, image1, image2)

    # use rich feature matching to compute point correspondences
    if choice == 'features':
        print "    Detecting keypoints...\n    Computing descriptors..."
        kp1, des1 = find_keypoints_descriptors(img1_gray)
        kp2, des2 = find_keypoints_descriptors(img2_gray)
        print "    Matching keypoints..."
        src_pts, dst_pts = match_keypoints(kp1, des1, kp2, des2)
        norm_pts1, norm_pts2 = normalize_points(K_matrices, src_pts, dst_pts)

        print "    Finding the essential and projection matrices..."
        E, mask = find_essential_matrix(K_matrices, norm_pts1, norm_pts2)
        P1, P2 = find_projection_matrices(E, poses)
        src_pts, dst_pts = filter_keypoints(mask, src_pts, dst_pts)
        norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
        refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

        print "    Triangulating 3D points..."
        homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
        norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)

        print "    Extracting colour information..."
        img1_pts, img2_pts, img_colours = get_colours(img1, K_matrices, norm_pts1, norm_pts2)

        print "    Initializing feature tracks..."
        pt_cloud_indexed = attach_indices(i, pts_3D, src_pts, dst_pts)

        return sensor_i, K_matrices, dst_pts, homog_3D, pts_3D, img_colours, pt_cloud_indexed

    # use Farneback's dense optical flow tracking to compute point correspondences
    elif choice == 'flow':
        while img1.shape[0] > 700 or img1.shape[1] > 700:
            img1, img2 = downsample_images(img1, img2)
            img1_gray, img2_gray = gray_downsampled(img1, img2)
        print "    Calculating optical flow..."
        flow = calc_flow(img1_gray, img2_gray)
        print "    Matching pixel coordinates..."
        src_pts, dst_pts = match_points(img2_gray, flow)
        norm_pts1, norm_pts2 = normalize_points(K_matrices, src_pts, dst_pts)

        print "    Finding the essential and projection matrices..."
        E, mask = find_essential_matrix(K_matrices, norm_pts1, norm_pts2)
        P1, P2 = find_projection_matrices(E, poses)
        norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
        refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

        print "    Triangulating 3D points..."
        homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
        norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)
        pts_3D, norm_pts1, norm_pts2 = filter_outliers(pts_3D, norm_pts1, norm_pts2)

        print "    Extracting colour information..."
        img1_pts, img2_pts, img_colours = get_colours(img1, K_matrices, norm_pts1, norm_pts2)

        print "    Initializing flow tracks..."
        pt_cloud_indexed = attach_tracks(i, pts_3D, norm_pts1, norm_pts2)

        return sensor_i, K_matrices, homog_3D, pts_3D, img_colours, pt_cloud_indexed

def find_new_pts_feat(i, prev_sensor, K_matrices, image1, image2, prev_dst, poses, pt_cloud_indexed, last):
    print "    Loading images..."
    img1, img2 = load_images(image1, image2)
    img1_gray, img2_gray = gray_images(img1, img2)
    print "    Building camera calibration matrices..."
    sensor_i, K_matrices = build_calibration_matrices(i, prev_sensor, K_matrices, image1, image2)

    print "    Detecting keypoints...\n    Computing descriptors..."
    prev_kp, prev_des = find_keypoints_descriptors(img1_gray)
    new_kp, new_des = find_keypoints_descriptors(img2_gray)
    print "    Matching keypoints..."
    src_pts, dst_pts = match_keypoints(prev_kp, prev_des, new_kp, new_des)
    norm_pts1, norm_pts2 = normalize_points(K_matrices, src_pts, dst_pts)

    E, mask = find_essential_matrix(K_matrices, norm_pts1, norm_pts2)
    src_pts, dst_pts = filter_keypoints(mask, src_pts, dst_pts)
    norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
    refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

    print "    Scanning cloud..."
    matched_pts_2D, matched_pts_3D, indices = scan_cloud(i, prev_dst, src_pts, pt_cloud_indexed)
    print "    Computing camera pose..."
    poses = compute_cam_pose(K_matrices, matched_pts_2D, matched_pts_3D, poses)
    P1, P2 = find_projection_matrices(E, poses)

    print "    Triangulating 3D points..."
    homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
    norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)

    print "    Extracting colour information..."
    img1_pts, img2_pts, img_colours = get_colours(img1, K_matrices, norm_pts1, norm_pts2)
    print "    Assembling feature tracks..."
    pt_cloud_indexed = attach_indices(i, pts_3D, src_pts, dst_pts, pt_cloud_indexed)

    if last:
        # find the pose of the last camera
        matched_pts_2D = np.array([ dst_pts[i] for i in indices ])
        poses = compute_cam_pose(K_matrices, matched_pts_2D, matched_pts_3D, poses)

    return sensor_i, K_matrices, dst_pts, poses, homog_3D, pts_3D, img_colours, pt_cloud_indexed


def find_new_pts_flow(i, prev_sensor, K_matrices, image1, image2, poses, pt_cloud_indexed):
    print "    Loading images..."
    img1, img2 = load_images(image1, image2)
    while img1.shape[0] > 700 or img1.shape[1] > 700:
        img1, img2 = downsample_images(img1, img2)
        img1_gray, img2_gray = gray_downsampled(img1, img2)
    print "    Building camera calibration matrices..."
    sensor_i, K_matrices = build_calibration_matrices(i, prev_sensor, K_matrices, image1, image2)

    print "    Calculating optical flow..."
    flow = calc_flow(img1_gray, img2_gray)
    print "    Matching pixel coordinates..."
    src_pts, dst_pts = match_points(img2_gray, flow)
    norm_pts1, norm_pts2 = normalize_points(K_matrices, src_pts, dst_pts)

    E, mask = find_essential_matrix(K_matrices, norm_pts1, norm_pts2)
    norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
    refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

    print "    Scanning cloud..."
    matched_pts_2D, matched_pts_3D = scan_tracks(i, norm_pts1, norm_pts2, pt_cloud_indexed)
    print "    Computing camera pose..."
    poses = compute_cam_pose(K_matrices, matched_pts_2D, matched_pts_3D, poses)
    P1, P2 = find_projection_matrices(E, poses)

    print "    Triangulating 3D points..."
    homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
    norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)

    print "    Extracting colour information..."
    img1_pts, img2_pts, img_colours = get_colours(img1, K_matrices, norm_pts1, norm_pts2)

    print "    Assembling flow tracks..."
    pt_cloud_indexed = attach_tracks(i, pts_3D, norm_pts1, norm_pts2)

    return sensor_i, K_matrices, poses, homog_3D, pts_3D, img_colours, pt_cloud_indexed


def start(choice, images, file_path=None):
    '''Loop through each pair of images, find point correspondences and generate 3D point cloud.
    For each new frame, find additional points and add them to the overall point cloud.'''
    prev_sensor = 0
    K_matrices = []
    poses = [np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)]
    save_format = 'txt'

    for i in range(len(images)-1):
        print "\n  Processing image %d and %d... " % (i+1, i+2)

        if choice == 'features':
            if i == 0:
                # first 2 images
                prev_sensor, K_matrices, prev_dst, homog_3D, pts_3D, img_colours, pt_cloud_indexed = gen_pt_cloud(i, prev_sensor, K_matrices, images[i], images[i+1], poses, choice)
                pt_cloud = np.array(pts_3D)
                colours = np.array(img_colours)
            elif i >= 1 and i != len(images)-2:
                prev_sensor, K_matrices, prev_dst, poses, homog_3D, pts_3D, img_colours, pt_cloud_indexed = find_new_pts_feat(i, prev_sensor, K_matrices, images[i], images[i+1], prev_dst, poses, pt_cloud_indexed, last=False)
                pt_cloud = np.vstack((pt_cloud, pts_3D))
                colours = np.vstack((colours, img_colours))
            elif i == len(images)-2:
                # last pair of images
                prev_sensor, K_matrices, prev_dst, poses, homog_3D, pts_3D, img_colours, pt_cloud_indexed = find_new_pts_feat(i, prev_sensor, K_matrices, images[i], images[i+1], prev_dst, poses, pt_cloud_indexed, last=True)
                pt_cloud = np.vstack((pt_cloud, pts_3D))
                colours = np.vstack((colours, img_colours))

        elif choice == 'flow':
            if i == 0:
                # first 2 frames
                prev_sensor, K_matrices, homog_3D, pts_3D, img_colours, pt_cloud_indexed = gen_pt_cloud(i, prev_sensor, K_matrices, images[i], images[i+1], poses, choice)
                pt_cloud = np.array(pts_3D)
                colours = np.array(img_colours)
            elif i >= 1:
                prev_sensor, K_matrices, poses, homog_3D, pts_3D, img_colours, pt_cloud_indexed = find_new_pts_flow(i, prev_sensor, K_matrices, images[i], images[i+1], poses, pt_cloud_indexed)
                pt_cloud = np.vstack((pt_cloud, pts_3D))
                colours = np.vstack((colours, img_colours))

    if choice == 'features':
        save_points(choice, images, pt_cloud, colours, file_path, save_format='pcd')
        save_points(choice, images, pt_cloud, colours, file_path, save_format='txt')
        print "    Removing outliers..."
        remove_outliers(file_path + '.pcd')
        pcd_to_txt(file_path + '.txt', file_path + '_inliers' + '.pcd')
        load_points(file_path + '_inliers.' + save_format)
        write_points_ba(pt_cloud_indexed, len(images), K_matrices, poses)

    elif choice == 'flow':
        save_points(choice, images, pt_cloud, colours, file_path, save_format='pcd')
        save_points(choice, images, pt_cloud, colours, file_path, save_format='txt')
        load_points(file_path + '.' + save_format)

    # homog_pt_cloud = np.vstack((pt_cloud.T, np.ones(pt_cloud.shape[0])))
    # draw.draw_matches(src_pts, dst_pts, img1_gray, img2_gray)
    # draw.draw_epilines(src_pts, dst_pts, img1_gray, img2_gray, F, mask)
    # draw.draw_projected_points(homog_pt_cloud, P)
    # display_vtk.vtk_show_points(pt_cloud, list(colours))

def main():
    choice = sys.argv[1]
    load_filename = ''#points/alcatraz1_inliers.txt'
    if load_filename != '':
        load_points(load_filename)
    else:
        directory = 'images/statue'
        # images = ['images/statue/P1000965.JPG', 'images/statue/P1000969.JPG']
        images = ['images/kermit/kermit001.jpg', 'images/kermit/kermit002.jpg', 'images/kermit/kermit003.jpg', 'images/kermit/kermit004.jpg']
        # images = ['images/ucd_building4_all/00000000.jpg', 'images/ucd_building4_all/00000002.jpg', 'images/ucd_building4_all/00000003.jpg']
        # images = ['images/ucd_coffeeshack_all/00000007.JPG', 'images/ucd_coffeeshack_all/00000008.JPG', 'images/ucd_coffeeshack_all/00000009.JPG', 'images/ucd_coffeeshack_all/00000010.JPG']
        # images = sort_images(directory)
        save_filepath = 'points/' + images[0].rpartition('/')[2].rpartition('.')[0]
        start(choice, images[:2], file_path=save_filepath)


if __name__ == "__main__":
    main()