from flask import Flask, flash, request, redirect, url_for, render_template
from utils.DHC_OCR import DHC_OCR
import os
from werkzeug.utils import secure_filename

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
        #print('upload_image filename: ' + filename)
        flash('Image successfully uploaded and displayed below')
        # print("\n\n\n Path: ", os.path.join(app.config['UPLOAD_FOLDER'], filename))
        # image = request.files['file']
        prediction = DHC_OCR.prediction_img(src=filepath)
        # flash(prediction)
        return render_template('index.html', filename=filename, prediction=prediction)
    
    else:
        flash('Allowed Image types are - png, jpg, jpeg')
        return redirect(request.url)
 
@app.route('/display/<filename>')
def display_image(filename):
    #print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

# @app.route('/display/<filename>', methods=['POST'])
# def predict(filename):
#     if 'file' not in request.files:
#         flash('No file part')
#         return redirect(request.url)
    
#     image = request.files['image']
#     ocr_model = DHC_OCR()
#     prediction = ocr_model.prediction(img=image)
#     print("pred", prediction)
#     flash(prediction)
#     return prediction

if __name__ == '__main__':
    app.run(debug=True)