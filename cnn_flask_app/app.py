from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
#import tensorflow as tf
from PIL import Image
import cv2

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['PROCESSED_FOLDER'] = 'static/processed'


model = load_model('step5-CNN-softmax-4_level_conv-15_epoch.keras', compile=False)


CATEGORIES = ["Cat", "Dog"]



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    IMG_SIZE = 75

    file = request.files['file']
    if not file:
        return redirect(url_for('home'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Ladda originalbilden och skala ner + konvertera till gråskala
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_reshaped = img_resized.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    img_normed = img_reshaped/255.0     # Normera gråskallenivå, (0,255) transformeras till (0,1), utan medlevärdesförskjutning

    processed_filename = 'gray_' + filename
    processed_path = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
    cv2.imwrite(processed_path, img_resized)
    
    result = model.predict(img_normed)
    prediction = CATEGORIES[np.argmax(result[0])]
    prob_cat = result[0][0]
    prob_dog = result[0][1]


    return render_template(
        'result.html',
        original_image=url_for('static', filename='uploads/' + filename),
        processed_image=url_for('static', filename='processed/' + processed_filename),
        prediction=prediction,
        prob_cat=prob_cat,
        prob_dog=prob_dog
    )



if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)
    app.run(debug=True)



