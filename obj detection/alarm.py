
import cv2
import numpy as np
import winsound  # For playing sound on Windows

# Load YOLO
yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

# Load classes
classes = []
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

# Get output layer names
layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)

# Open webcam
cap = cv2.VideoCapture(0)  # Use 0 for default webcam, or change to the appropriate index if you have multiple webcams

# Initialize alarm parameters
alarm_active = False
alarm_duration = 500  # in milliseconds
alarm_frequency = 1500  # in Hertz

while True:
    # Read frame from webcam
    ret, frame = cap.read()

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    yolo.setInput(blob)
    outputs = yolo.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                w = int(detection[2] * frame.shape[1])
                h = int(detection[3] * frame.shape[0])

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Check if a person is detected
    alarm_active = any(classes[class_ids[i]] == 'person' for i in indexes)

    # Draw rectangles and display labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = confidences[i]

            # Draw rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), colorGreen, 3)

            # Display class label and confidence
            text = f"{label}: {confidence:.2f}"
            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorRed, 2)

    # Check if the alarm should be active
    if alarm_active:
        # Play the alarm sound
        winsound.Beep(alarm_frequency, alarm_duration)

    # Display the frame
    cv2.imshow("Webcam", frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
