from flask import Flask, render_template, Response
import cv2
import numpy as np
import winsound
import sqlite3
import datetime

app = Flask(__name__)

# Create SQLite database
conn = sqlite3.connect('detected_objects.db')
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS detected_objects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT,
        confidence REAL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
''')
conn.commit()
conn.close()

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

# Initialize alarm parameters
alarm_duration = 500  # in milliseconds
alarm_frequency = 1500  # in Hertz

# Function to generate video frames for the webcam feed
def generate_frames():
    cap = cv2.VideoCapture(0)

    while True:
        success, frame = cap.read()
        if not success:
            break

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
                    label = str(classes[class_id])
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

                    # Save the detected object to the database
                    conn = sqlite3.connect('detected_objects.db')
                    cursor = conn.cursor()
                    cursor.execute('''
                        INSERT INTO detected_objects (label, confidence) VALUES (?, ?)
                    ''', (label, confidence))
                    conn.commit()
                    conn.close()

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = confidences[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), colorGreen, 3)
                text = f"{label}: {confidence:.2f}"
                cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorRed, 2)

        if any(classes[class_ids[i]] == 'person' for i in indexes):
            winsound.Beep(alarm_frequency, alarm_duration)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route to render the HTML page
@app.route('/')
def index():
    return render_template('index.html')

# Route to stream the video feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)