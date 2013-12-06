## Reconstruct

RECONSTRUCT automatically generates 3D point clouds from 2 or more photos using either a robust local feature detector ([SURF](http://www.vision.ee.ethz.ch/~surf/eccv06.pdf)) or dense optical flow ([Farneback's algorithm](http://lmi.bwh.harvard.edu/papers/pdfs/gunnar/farnebackICPR00.pdf)) to recover the "missing" depth information and establish feature tracks across different frames. Technologies used are Python, NumPy, SciPy, OpenCV, Flask, SQLAlchemy, SQLite, Postgres, Javascript, jQuery, AJAX, JSON, ThreeJS, HTML5 and CSS3.

### 3D reconstruction
###### (reconstruct.py)

Overview of the structure from motion pipeline that generates a point cloud from the first pair of images, and finds additional points to be added from each subsequent image. Calls functions from processing.py, rich_features.py and optical_flow.py to do the actual reconstruction.

### Find point correspondences
###### (rich_features.py & optical_flow.py)

2 algorithms are available for finding matching points between images, SURF and dense optical flow. 
The SURF algorithm detects specific features (blobs) in each image and attaches a descriptor to each feature based on the sum of the Haar wavelet responses around each point of interest. The descriptors are then compared to obtain the nearest neighbour matches. Because only specific features are detected and matched, the resulting point cloud is sparse
Farneback's dense optical flow algorithm, normally used for motion tracking in video, is available as an option for generating a denser point cloud (albeit under stricter constraints). The algorithm uses local polynomial expansion to estimate displacement fields for every pixel in the image, thereby relating corresponding pixels in a pair of images.

### Image processing
###### (processing.py)

Pre-processes images prior to the reconstruction by graying out the images and downsampling them to an appropriate size if they are too large. The bulk of the reconstruction is done here, save for algorithm-specific functions above. The reconstruction algorithm builds the camera calibration matrices by extracting information from the EXIF metadata embedded in the images, computes the essential and projection matrices, and triangulates 3D points from the 2D point correspondences using Direct Linear Transform. For the third and subsequent images in the sequence, it also computes the pose of each new camera in the scene so that the new 3D points to be added are in the same coordinate system as that of the first camera (which is taken to be the frame of reference).

### Reconstruction tests
###### (draw.py)

To ensure that the reconstructions is valid, a number of functions were written to visualize the results of the intermediate steps, such as the point correspondences, epilines and 2D projection of the 3D points (this was written before the 3D viewer was completed).

### Databases
###### (model.py & cam_db.py)

The main SQLite database stores all user-related information, including user-uploaded photos and user-generated point clouds. A smaller Postgres database stores camera sensor sizes required for building the camera calibration matrices during reconstruction.

### Web Framework
###### (views.py)

Flask is used as the web framework. Users upload 2 or more photos, and the data URLs of the selected photos are sent via AJAX to the server, where the route that returns the text file of the generated point cloud is called. Logins and registration are also handled here along with all other communications with the main database to add or retrieve information.

### User interface
###### (interface.js - in static/js)

jQuery is used to ensure a smooth user experience in the single-page webapp, from uploading to viewing and downloading. The drag-and-drop interface built in Javascript allows users to intuitively upload photos in the browser. If users are logged in, their previous point clouds are also automatically retrieved from the database and sent to the browser in JSON format via AJAX to be displayed.

### 3D viewer
###### (viewer.js - in static/js)

The 3D viewer is built using THREEjs. It loads the text file generated from the reconstruction, and parses it line-by-line to extract the 3D coordinates of each vertex, which is then added to the point cloud.

### Screenshots

![Point cloud library](/screenshots/RECONSTRUCT1.png)
![Upload photos](/screenshots/RECONSTRUCT2.png)
![View point cloud](/screenshots/RECONSTRUCT3.png)