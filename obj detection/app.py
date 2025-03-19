from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import winsound
import sqlite3
import datetime

app = Flask(__name__)

yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

classes = []
with open("coco.names", "r") as file:
    classes = [line.strip() for line in file.readlines()]

layer_names = yolo.getLayerNames()
output_layers = [layer_names[i - 1] for i in yolo.getUnconnectedOutLayers()]

colorRed = (0, 0, 255)
colorGreen = (0, 255, 0)

conn = sqlite3.connect('detected_objects.db', check_same_thread=False)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS detected_objects (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        label TEXT,
        timestamp TEXT
    )
''')
conn.commit()

running = False

def generate_frames():
    global running
    cap = cv2.VideoCapture(0)

    while running:
        success, frame = cap.read()
        if not success:
            break

        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        yolo.setInput(blob)
        outputs = yolo.forward(output_layers)

        class_ids = []
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
                    class_ids.append(class_id)

                    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    cursor.execute('INSERT INTO detected_objects (label, timestamp) VALUES (?, ?)', (label, current_time))
                    conn.commit()

        indexes = cv2.dnn.NMSBoxes(boxes, [1.0] * len(boxes), 0.5, 0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                cv2.rectangle(frame, (x, y), (x + w, y + h), colorGreen, 3)
                cv2.putText(frame, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colorRed, 2)

        if any(classes[class_ids[i]] == 'person' for i in indexes):
            winsound.Beep(1500, 500)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    global running
    running = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop_feed')
def stop_feed():
    global running
    running = False
    return jsonify({"message": "Video feed stopped"})

@app.route('/get_detected_objects')
def get_detected_objects():
    cursor.execute('SELECT label, timestamp FROM detected_objects ORDER BY id DESC LIMIT 5')
    rows = cursor.fetchall()
    objects = [{"label": row[0], "timestamp": row[1]} for row in rows]
    return jsonify({"objects": objects})

if __name__ == '__main__':
    app.run(debug=True)
