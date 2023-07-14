import cv2
import numpy as np
import open3d as o3d
from scipy.ndimage import uniform_filter
import os


#
# Load YOLO
net = cv2.dnn.readNet("YOLO_files/yolov3.weights", "YOLO_files/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = net.getUnconnectedOutLayersNames()

# Loading the labels
classes = []
with open("YOLO_files/coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Classes for Bicycle, Car, Motorcycle, Bus, and Truck
selected_classes = ['bicycle', 'car', 'motorcycle', 'bus', 'truck']

# Loading video
cap = cv2.VideoCapture('video_sources/traf_video_Trim.mp4')
width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)  # float
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # float

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
output_video = cv2.VideoWriter('output.mp4', fourcc, 20.0, (int(width), int(height)))

while(cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break

    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and classes[class_id] in selected_classes:  # Only select classes specified
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            cv2.rectangle(img, (x, y), (x + w, y + h), (0,255,0), 2)
            cv2.putText(img, label, (x, y + 30), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0), 2)

    # only write frame to video file if img is not None
    if img is not None:
        output_video.write(img)

cap.release()
output_video.release()
cv2.destroyAllWindows()


# Load the video and get the first frame
cap = cv2.VideoCapture(r"video_sources\traf_video_Trim.mp4")
ret, frame1 = cap.read()

# Check if the first frame was successfully read
if not ret:
    print("Error reading the first frame from the video.")
    exit(1)

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

    """
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
    """

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




# pip install opencv-python
# pip install open3d
# pip install scipy
# pip install numpy
