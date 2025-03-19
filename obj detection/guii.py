import cv2
import numpy as np
import winsound
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk

class ObjectDetectionApp:
    def __init__(self, root, video_source=0):
        self.root = root
        self.root.title("Object Detection App")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.video_source = video_source
        self.cap = cv2.VideoCapture(self.video_source)

        self.label_title = ttk.Label(root, text="Object Detection App", font=("Helvetica", 16, "bold"))
        self.label_title.grid(row=0, column=0, columnspan=2, pady=(10, 20))

        self.canvas = tk.Canvas(root, width=640, height=480)
        self.canvas.grid(row=1, column=0, columnspan=2)

        self.label_status = ttk.Label(root, text="Status: Ready", font=("Helvetica", 12))
        self.label_status.grid(row=2, column=0, columnspan=2, pady=(0, 10))

        self.btn_start = ttk.Button(root, text="Start Detection", command=self.start_detection)
        self.btn_start.grid(row=3, column=0, padx=10, pady=10)

        self.btn_stop = ttk.Button(root, text="Stop Detection", command=self.stop_detection)
        self.btn_stop.grid(row=3, column=1, padx=10, pady=10)

        self.alarm_active = False
        self.alarm_duration = 500
        self.alarm_frequency = 1500

        self.yolo = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        self.classes = []
        with open("coco.names", "r") as file:
            self.classes = [line.strip() for line in file.readlines()]
        self.layer_names = self.yolo.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.yolo.getUnconnectedOutLayers()]

        self.detect_objects()

    def start_detection(self):
        self.alarm_active = True
        self.label_status.config(text="Status: Detection Active")
        self.detect_objects()

    def stop_detection(self):
        self.alarm_active = False
        self.label_status.config(text="Status: Detection Stopped")

    def on_close(self):
        self.cap.release()
        self.root.destroy()

    def play_alarm(self):
        winsound.Beep(self.alarm_frequency, self.alarm_duration)

    def detect_objects(self):
        ret, frame = self.cap.read()

        if ret:
            frame = self.detect_objects_in_frame(frame)

            if self.alarm_active and self.object_detected(frame):
                self.play_alarm()

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)

            if self.alarm_active:
                self.root.after(10, self.detect_objects)  # Continue detection

    def object_detected(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.yolo.setInput(blob)
        outputs = self.yolo.forward(self.output_layers)

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Check if the detected object is a person (class_id 0 corresponds to "person" in the COCO dataset)
                if class_id == 0 and confidence > 0.5:
                    return True

        return False

    def detect_objects_in_frame(self, frame):
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        self.yolo.setInput(blob)
        outputs = self.yolo.forward(self.output_layers)

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                # Check if the detected object is a person (class_id 0 corresponds to "person" in the COCO dataset)
                if class_id == 0 and confidence > 0.5:
                    center_x = int(detection[0] * frame.shape[1])
                    center_y = int(detection[1] * frame.shape[0])
                    w = int(detection[2] * frame.shape[1])
                    h = int(detection[3] * frame.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                    text = f"Person: {confidence:.2f}"
                    cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        return frame

if __name__ == "__main__":
    root = tk.Tk()
    app = ObjectDetectionApp(root)
    root.mainloop()
