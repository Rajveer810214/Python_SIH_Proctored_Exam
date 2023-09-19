
import cv2
import numpy as np
import face_recognition
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route("/compare", methods=["POST"])
def compare_images():
    try:
        # Get the image buffers from the request
        image1_data = request.files['image1'].read()
        image2_data = request.files['image2'].read()

        # Decode the image data using OpenCV
        image1 = cv2.imdecode(np.frombuffer(image1_data, np.uint8), cv2.IMREAD_COLOR)
        image2 = cv2.imdecode(np.frombuffer(image2_data, np.uint8), cv2.IMREAD_COLOR)

        # Load a reference face (you should replace this with your own database of known faces)
        known_face_image = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        known_face_encodings = face_recognition.face_encodings(known_face_image)

        if not known_face_encodings:
            return "No face found in the reference image"

        known_face_encoding = known_face_encodings[0]

        # Check if any faces are detected in the uploaded image
        uploaded_face_image = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)  # Convert to RGB format
        uploaded_face_encodings = face_recognition.face_encodings(uploaded_face_image)
        
        if not uploaded_face_encodings:
            return "No face found in the uploaded image"

        # Compare the detected face with the known face encoding
        match = face_recognition.compare_faces(known_face_encodings, uploaded_face_encodings[0], tolerance=0.6)

        if any(match):
            return "Face matched! Access granted."
        else:
            return "Face not recognized. Access denied."
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
