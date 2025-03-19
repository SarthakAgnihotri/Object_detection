from flask import Flask, render_template, Response
import cv2
import base64

app = Flask(__name__)

# Function to generate video frames from the webcam
def generate_frames():
    cap = cv2.VideoCapture(0)  # 0 corresponds to the default camera
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Perform your object detection or processing on the 'frame' here
        # For example, you can use the OpenCV functions to draw rectangles, text, etc.

        # Encode the frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer).decode('utf-8')

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_encoded.encode() + b'\r\n')

    cap.release()

# Route to render the HTML template
@app.route('/')
def index():
    return render_template('index.html')

# Route to serve the webcam feed
@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
