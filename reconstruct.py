import sys
import numpy as np
import draw, display_vtk
from processing import *
from rich_features import *
from optical_flow import *

def gen_pt_cloud(i, prev_sensor, image1, image2, poses, choice):
    '''Generates a point cloud for a pair of images.'''
    print "    Loading images..."
    img1, img2 = load_images(image1, image2)
    img1_gray, img2_gray = gray_images(img1, img2)
    print "    Building camera calibration matrices..."
    sensor_i, K1, K2 = build_calibration_matrices(i, prev_sensor, image1, image2)

    # use rich feature matching to compute point correspondences
    if choice == 'features':
        print "    Detecting keypoints...\n    Computing descriptors..."
        kp1, des1 = find_keypoints_descriptors(img1_gray)
        kp2, des2 = find_keypoints_descriptors(img2_gray)
        print "    Matching keypoints..."
        src_pts, dst_pts, filtered_kp1, filtered_kp2 = match_keypoints(kp1, des1, kp2, des2)
        norm_pts1, norm_pts2 = normalize_points(K1, K2, src_pts, dst_pts)

        print "    Finding the essential and projection matrices..."
        E, mask = find_essential_matrix(K1, norm_pts1, norm_pts2)
        P1, P2 = find_projection_matrices(E, poses)
        filtered_kp1, filtered_kp2 = filter_keypoints(mask, filtered_kp1, filtered_kp2)
        norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
        refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

        print "    Triangulating 3D points..."
        homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
        norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)

        print "    Extracting colour information..."
        img1_pts, img2_pts, img_colours = get_colours(img1, K1, K2, norm_pts1, norm_pts2, homog_3D)

        print "    Initializing feature tracks..."
        pt_cloud_indexed = attach_indices(i, pts_3D, filtered_kp1, filtered_kp2)

        return sensor_i, kp2, des2, filtered_kp2, homog_3D, pts_3D, img_colours, pt_cloud_indexed

    # use Farneback's dense optical flow tracking to compute point correspondences
    elif choice == 'flow':
        while img1.shape[0] > 1000 or img1.shape[1] > 1000:
            img1, img2 = downsample_images(img1, img2)
            img1_gray, img2_gray = gray_downsampled(img1, img2)
        print "    Calculating optical flow..."
        flow = calc_flow(img1_gray, img2_gray)
        print "    Matching pixel coordinates..."
        src_pts, dst_pts = match_points(img2_gray, flow)
        norm_pts1, norm_pts2 = normalize_points(K1, K2, src_pts, dst_pts)

        print "    Finding the essential and projection matrices..."
        E, mask = find_essential_matrix(K1, norm_pts1, norm_pts2)
        P1, P2 = find_projection_matrices(E, poses)
        norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
        refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

        print "    Triangulating 3D points..."
        homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
        norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)

        print "    Extracting colour information..."
        img1_pts, img2_pts, img_colours = get_colours(img1, K1, K2, norm_pts1, norm_pts2, homog_3D)

        print "    Initializing flow tracks..."
        pt_cloud_indexed = attach_tracks(i, pts_3D, norm_pts1, norm_pts2)

        return sensor_i, homog_3D, pts_3D, img_colours, pt_cloud_indexed

def find_new_pts_feat(i, prev_sensor, image1, image2, prev_kp, prev_des, prev_filter, poses, pt_cloud_indexed):
    print "    Loading images..."
    img1, img2 = load_images(image1, image2)
    img1_gray, img2_gray = gray_images(img1, img2)
    print "    Building camera calibration matrices..."
    sensor_i, K1, K2 = build_calibration_matrices(i, prev_sensor, image1, image2)

    print "    Detecting keypoints...\n    Computing descriptors..."
    new_kp, new_des = find_keypoints_descriptors(img2_gray)
    print "    Matching keypoints..."
    src_pts, dst_pts, filtered_kp1, filtered_kp2 = match_keypoints(prev_kp, prev_des, new_kp, new_des)
    norm_pts1, norm_pts2 = normalize_points(K1, K2, src_pts, dst_pts)

    E, mask = find_essential_matrix(K1, norm_pts1, norm_pts2)
    filtered_kp1, filtered_kp2 = filter_keypoints(mask, filtered_kp1, filtered_kp2)
    norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
    refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

    print "    Scanning cloud..."
    matched_pts_2D, matched_pts_3D = scan_cloud(i, prev_kp, prev_des, prev_filter, filtered_kp1, pt_cloud_indexed)
    print "    Computing camera pose..."
    poses = compute_cam_pose(K1, matched_pts_2D, matched_pts_3D, poses)
    P1, P2 = find_projection_matrices(E, poses)

    print "    Triangulating 3D points..."
    homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
    norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)

    print "    Extracting colour information..."
    img1_pts, img2_pts, img_colours = get_colours(img1, K1, K2, norm_pts1, norm_pts2, homog_3D)
    print "    Assembling feature tracks..."
    pt_cloud_indexed = attach_indices(i, pts_3D, filtered_kp1, filtered_kp2, pt_cloud_indexed)

    return sensor_i, new_kp, new_des, filtered_kp2, poses, homog_3D, pts_3D, img_colours, pt_cloud_indexed


def find_new_pts_flow(i, prev_sensor, image1, image2, poses, pt_cloud_indexed):
    print "    Loading images..."
    img1, img2 = load_images(image1, image2)
    while img1.shape[0] > 1000 or img1.shape[1] > 1000:
        img1, img2 = downsample_images(img1, img2)
        img1_gray, img2_gray = gray_downsampled(img1, img2)
    print "    Building camera calibration matrices..."
    sensor_i, K1, K2 = build_calibration_matrices(i, prev_sensor, image1, image2)

    print "    Calculating optical flow..."
    flow = calc_flow(img1_gray, img2_gray)
    print "    Matching pixel coordinates..."
    src_pts, dst_pts = match_points(img2_gray, flow)
    norm_pts1, norm_pts2 = normalize_points(K1, K2, src_pts, dst_pts)

    E, mask = find_essential_matrix(K1, norm_pts1, norm_pts2)
    norm_pts1, norm_pts2 = apply_mask(mask, norm_pts1, norm_pts2)
    refined_pts1, refined_pts2 = refine_points(norm_pts1, norm_pts2, E)

    print "    Scanning cloud..."
    matched_pts_2D, matched_pts_3D = scan_tracks(i, norm_pts1, norm_pts2, pt_cloud_indexed)
    print "    Computing camera pose..."
    poses = compute_cam_pose(K1, matched_pts_2D, matched_pts_3D, poses)
    P1, P2 = find_projection_matrices(E, poses)

    print "    Triangulating 3D points..."
    homog_3D, pts_3D, infront = triangulate_points(P1, P2, refined_pts1, refined_pts2)
    norm_pts1, norm_pts2 = apply_infront_filter(infront, norm_pts1, norm_pts2)

    print "    Extracting colour information..."
    img1_pts, img2_pts, img_colours = get_colours(img1, K1, K2, norm_pts1, norm_pts2, homog_3D)

    print "    Assembling flow tracks..."
    pt_cloud_indexed = attach_tracks(i, pts_3D, norm_pts1, norm_pts2)

    return sensor_i, poses, homog_3D, pts_3D, img_colours, pt_cloud_indexed


def start(choice, images, file_path=None):
    '''Loop through each pair of images, find point correspondences and generate 3D point cloud.
    For each new frame, find additional points and add them to the overall point cloud.'''
    prev_sensor = 0
    poses = [np.array([[1,0,0,0], [0,1,0,0], [0,0,1,0]], dtype=float)]
    save_format = 'txt'

    for i in range(len(images)-1):
        print "\n  Processing image %d and %d... " % (i+1, i+2)

        if choice == 'features':
            if i == 0:
                # first 2 images
                prev_sensor, prev_kp, prev_des, prev_filter, homog_3D, pts_3D, img_colours, pt_cloud_indexed = gen_pt_cloud(i, prev_sensor, images[i], images[i+1], poses, choice)
                pt_cloud = np.array(pts_3D)
                colours = np.array(img_colours)
            elif i >= 1:
                prev_sensor, prev_kp, prev_des, prev_filter, poses, homog_3D, pts_3D, img_colours, pt_cloud_indexed = find_new_pts_feat(i, prev_sensor, images[i], images[i+1], prev_kp, prev_des, prev_filter, poses, pt_cloud_indexed)
                pt_cloud = np.vstack((pt_cloud, pts_3D))
                colours = np.vstack((colours, img_colours))

        elif choice == 'flow':
            if i == 0:
                # first 2 frames
                prev_sensor, homog_3D, pts_3D, img_colours, pt_cloud_indexed = gen_pt_cloud(i, prev_sensor, images[i], images[i+1], poses, choice)
                pt_cloud = np.array(pts_3D)
                colours = np.array(img_colours)
                print len(pt_cloud), len(colours)
            elif i >= 1:
                prev_sensor, poses, homog_3D, pts_3D, img_colours, pt_cloud_indexed = find_new_pts_flow(i, prev_sensor, images[i], images[i+1], poses, pt_cloud_indexed)
                pt_cloud = np.vstack((pt_cloud, pts_3D))
                colours = np.vstack((colours, img_colours))

    if choice == 'features':
        save_points(choice, images, pt_cloud, colours, file_path, save_format='pcd')
        save_points(choice, images, pt_cloud, colours, file_path, save_format='txt')
        print "    Removing outliers..."
        remove_outliers(file_path + '.pcd')
        pcd_to_txt(file_path + '.txt', file_path + '_inliers' + '.pcd')

    elif choice == 'flow':
        save_points(choice, images, pt_cloud, colours, file_path, save_format='pcd')
        save_points(choice, images, pt_cloud, colours, file_path, save_format='txt')

    # homog_pt_cloud = np.vstack((pt_cloud.T, np.ones(pt_cloud.shape[0])))
    # draw.draw_matches(src_pts, dst_pts, img1_gray, img2_gray)
    # draw.draw_epilines(src_pts, dst_pts, img1_gray, img2_gray, F, mask)
    # draw.draw_projected_points(homog_pt_cloud, P)

    # display_vtk.vtk_show_points(pt_cloud, list(colours))
    load_points(file_path + '.' + save_format)


def main():
    choice = sys.argv[1]
    load_filename = ''#points/alcatraz1_inliers.txt'

    if load_filename != '':
        load_points(load_filename)
    else:
        directory = 'images/ET'
        # images = ['images/kermit/kermit000.jpg', 'images/kermit/kermit003.jpg']
        # images = ['images/ucd_building4_all/00000000.jpg', 'images/ucd_building4_all/00000003.jpg']
        # images = ['images/ucd_coffeeshack_all/00000000.JPG', 'images/ucd_coffeeshack_all/00000003.JPG']
        images = sort_images(directory)
        save_filepath = 'points/' + images[0].rpartition('/')[2].rpartition('.')[0]
        start(choice, images[:4], file_path=save_filepath)


if __name__ == "__main__":
    main()