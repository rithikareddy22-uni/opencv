import os
import cv2
import numpy as np
from flask import Flask, render_template, request, Response

app = Flask(__name__, template_folder='../templates')

CASCADE_PATH = os.path.join(os.path.dirname(__file__), 'haarcascade_frontalface_default.xml')
face_cascade = cv2.CascadeClassifier(os.path.abspath(CASCADE_PATH))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/process_frame', methods=['POST'])
def process_frame():
    file = request.files.get('frame')
    if not file:
        return "No frame", 400

    # Convert uploaded frame bytes into OpenCV matrix
    file_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        return "Invalid frame", 400

    # Face detection and Gaussian blurring
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]
        blurred_face = cv2.GaussianBlur(face_roi, (99, 99), 30)
        img[y:y+h, x:x+w] = blurred_face

    # Encode back to JPEG to display in browser
    _, buffer = cv2.imencode('.jpg', img)
    return Response(buffer.tobytes(), mimetype='image/jpeg')

if __name__ == '__main__':
    app.run(debug=True)