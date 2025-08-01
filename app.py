from flask import Flask, render_template, request, redirect, url_for, Response
import cv2
import face_recognition
import sqlite3
import numpy as np

app = Flask(__name__)
video_capture = cv2.VideoCapture(0)

# === Load known faces from database ===
def load_known_faces():
    conn = sqlite3.connect('FaceTestDb')
    cursor = conn.cursor()
    cursor.execute("SELECT name, image FROM users")
    known_encodings = []
    known_names = []

    for name, image_blob in cursor.fetchall():
        np_image = np.frombuffer(image_blob, np.uint8)
        image = cv2.imdecode(np_image, cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        encodings = face_recognition.face_encodings(rgb)
        if encodings:
            known_encodings.append(encodings[0])
            known_names.append(name)

    conn.close()
    return known_encodings, known_names

known_encodings, known_names = load_known_faces()

@app.route('/register', methods=['POST'])
def register():
    action = request.form.get('action')

    if action == 'signup':
        name = request.form['name']
        image_file = request.files['image']

        if not name or not image_file:
            return "Name and Image required", 400

        image_data = image_file.read()

        conn = sqlite3.connect('FaceTestDb')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (name, image) VALUES (?, ?)", (name, image_data))
        conn.commit()
        conn.close()

        # Reload face data
        global known_encodings, known_names
        known_encodings, known_names = load_known_faces()

        return redirect('/video')
    
    return "Invalid action", 400
# === Generate video frames ===
def gen_frames():
    process_every_n = 3
    frame_count = 0
    face_locations = []
    face_encodings = []
    face_names = []

    while True:
        success, frame = video_capture.read()
        if not success:
            break

        frame_count += 1
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        if frame_count % process_every_n == 0:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_encodings, encoding)
                if face_distances.any():
                    best_match = np.argmin(face_distances)
                    if matches[best_match]:
                        name = known_names[best_match]
                face_names.append(name)

        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top, right, bottom, left = [v * 4 for v in (top, right, bottom, left)]
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# === Routes ===
@app.route('/')
def index():
    return render_template('register.html')

@app.route('/video')
def video():
    return render_template('index.html')  # Show camera page

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
