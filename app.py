from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('model/pneumonia_cnn_model.h5')

UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Class labels
classes = ['Bacterial Pneumonia','Normal', 'Viral Pneumonia']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or 'actual_class' not in request.form:
        return "Missing file or actual class"

    file = request.files['file']
    actual_class = request.form['actual_class']

    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Preprocess the image
    img = image.load_img(filepath, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict using the model
    prediction = model.predict(img_array)[0]
    predicted_class = classes[np.argmax(prediction)]

    return render_template(
        'index.html',
        result=predicted_class,
        actual=actual_class,
        img_path=filepath
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)
