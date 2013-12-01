# Reconstruct

RECONSTRUCT automatically generates 3D point clouds from 2 or more photos using either a robust local feature detector ([SURF](http://www.vision.ee.ethz.ch/~surf/eccv06.pdf)) or dense optical flow ([Farneback's algorithm](http://lmi.bwh.harvard.edu/papers/pdfs/gunnar/farnebackICPR00.pdf)) to recover the "missing" depth information and establish feature tracks across different frames. 

## Client-side

Users upload 2 or more photos anonymously or while logged in. Selected photos are sent to the server for processing, and the resultant point cloud is sent back to the client as JSON via AJAX. The point cloud is then loaded into a 3D viewer built on ThreeJS. Logged-in users can also save or download the reconstructed 3D point cloud.

Technologies used: Flask, Javascript, jQuery, Ajax, JSON, ThreeJS, HTML5, CSS3

## Server-side

The actual reconstruction is done mainly using Python and OpenCV, and encompasses camera autocalibration using EXIF metadata, keypoint detection and matching, camera pose estimation and triangulation of 3D points. Linear algebra and other matrix operations were computed using NumPy. A SQLite database stores all user-related information including user-uploaded photos and user-generated point clouds, while a smaller Postgres database stores camera sensor size information used in autocalibration.

Technologies used: Python, OpenCV, NumPy, SciPy, SQLite, PostgreSQL, SQLAlchemy

## Examples

[Sparse point cloud from feature matching](http://i.imgur.com/5710FTz.png)
[Dense point cloud from optical flow](http://i.imgur.com/sDdXn8h.png)

## Post-Hackbright goals

Complete bundle adjustment, mesh generation and texture mapping using SSBA and PCL (both C++ based). Extend functionality to include an option for 3D reconstruction from videos.