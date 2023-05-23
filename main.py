import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage.filters import uniform_filter

# Load the video and get the first frame
cap = cv2.VideoCapture(r"C:\Users\ipekd\ENS491-RealityCapture\traffic_2 - Made with Clipchamp.mp4")
ret, frame1 = cap.read()

# Create a FAST feature detector
fast = cv2.FastFeatureDetector_create(threshold=20, nonmaxSuppression=True)

# Create some empty arrays to store the points and their corresponding depth
p1 = np.empty((0, 2))
p2 = np.empty((0, 2))
depths = np.empty((0))

# Loop through the video frame by frame
while cap.isOpened():
    ret, frame2 = cap.read()

    # Break if there is no more video
    if not ret:
        break

    # Convert the frames to grayscale
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Detect keypoints using FAST in the first frame
    keypoints_1 = fast.detect(gray1, None)

    # Extract the keypoint locations
    p_0 = np.array([kp.pt for kp in keypoints_1], dtype=np.float32)

    # Calculate the optical flow between the first and second frames
    p_1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p_0, None)

    # Select only good points
    good_new = p_1[st[:, 0] == 1]
    good_old = p_0[st[:, 0] == 1]

    # Calculate the depth of each point based on the motion of the camera
    depth = np.abs(good_new - good_old)

    # Apply a moving average filter to the depth estimates
    window_size = 3  # or whatever window size you want
    depth = uniform_filter(depth, size=window_size, mode='reflect')

    # Append the good points and their depth to the arrays
    p2 = np.vstack([p2, good_new.reshape(-1, 2)])
    p1 = np.vstack([p1, good_old.reshape(-1, 2)])
    depths = np.append(depths, np.full((good_new.shape[0],), depth[:, 0].mean()))



    # Set the first frame to be the second frame for the next iteration
    frame1 = frame2.copy()

    # At the end of each loop iteration
    if len(good_old) > 0:
        # Draw the good features on the frame
        for i, point in enumerate(good_old):
            point = (int(point[0]), int(point[1]))
            cv2.circle(frame1, point, 5, (0, 0, 255), -1)

            # Draw the depth value next to the point
            depth = depths[i]
            cv2.putText(frame1, f'depth: {depth:.2f}', (point[0] + 5, point[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 255), 2, cv2.LINE_AA)

        # Resize and show the frame
        cv2.imshow('Feature Points', cv2.resize(frame1, (1800, int(frame1.shape[0] * 1800 / frame1.shape[1]))))

    # Break if the user presses the 'q' key
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

# Release the video capture object
cap.release()
cv2.destroyAllWindows()

# Convert the arrays to the correct format
p1 = np.round(p1).astype(int)
depths = depths[:, np.newaxis]

# Ensure p1 values do not exceed image dimensions
height, width, _ = frame1.shape
p1[:, 0] = np.clip(p1[:, 0], 0, width-1)
p1[:, 1] = np.clip(p1[:, 1], 0, height-1)

# Convert 2D points to 1D indices
indices = p1[:, 1] * width + p1[:, 0]

# Reshape the image to a 2D array of colors
colors = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)
colors = colors.reshape(-1, 3)

# Create a point cloud
pcd = o3d.geometry.PointCloud()

# Set the points and colors of the point cloud
pcd.points = o3d.utility.Vector3dVector(np.hstack((p1, depths)))
pcd.colors = o3d.utility.Vector3dVector(colors[indices][:, ::-1] / 255)

# Show the point cloud
o3d.visualization.draw_geometries([pcd])
