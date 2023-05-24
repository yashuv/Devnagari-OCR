from flask import Flask, flash, request, redirect, url_for, render_template
from utils.DHC_OCR import DHC_OCR
import os
from werkzeug.utils import secure_filename
import base64

# class_names = ['0', '1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '4', '40', '41', '42', '43', '44', '45', '5', '6', '7', '8', '9']

# labels_csv = pd.read_csv("labels.csv")
# # print(labels_csv)

# model = tf.keras.models.load_model('nepali_ocr_model.h5')

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "ocr"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
 
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/camera')
def index():
    return render_template('camera.html')
 
@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        flash('Image successfully uploaded and displayed below')

        print("xyz file path = ", filepath)
        docr = DHC_OCR()
        devanagari_label, success = docr.predict_image(img=filepath)
        # devanagari_label, success = docr.predict_webcam()

        output_txt = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(devanagari_label, success)
        print("yashuv app.py -->  {}".format(output_txt))
        docr.segment_prediction

        return render_template('index.html', filename=filename, prediction=output_txt)
    else:
        flash('Allowed Image types are - png, jpg, jpeg')
        return redirect(request.url)
 



@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

@app.route('/camera', methods=['POST'])
def upload_and_predict_image():
    image_data = request.form.get('image_data')
    if image_data:
        filename = secure_filename('captured_image.jpg')
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    
        with open(filepath, 'wb') as file:
            file.write(base64.b64decode(image_data.split(',')[1]))

        # Perform image prediction using the saved image
        docr = DHC_OCR()
        devanagari_label, success = docr.predict_image(img=filepath)
        # devanagari_label, success = docr.predict_webcam()

        output_txt = "This image most likely belongs to {} with a {:.2f} percent confidence.".format(devanagari_label, success)
        print("yashuv app.py -->  {}".format(output_txt))
        docr.segment_prediction
        return render_template('camera.html', pred=output_txt)

    else:
        flash('No image data found')
        return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)