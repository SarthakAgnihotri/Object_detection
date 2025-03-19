import cv2

# Check OpenCV version
print("OpenCV version:", cv2.__version__)

# Check if OpenCV was built with CUDA support
print("CUDA support:", cv2.cuda.getCudaEnabledDeviceCount() > 0)

# Create a VideoCapture object
cap = cv2.VideoCapture(0)

# Check if the VideoCapture object is opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
else:
    print("Webcam opened successfully.")

# Release the VideoCapture object
cap.release()
