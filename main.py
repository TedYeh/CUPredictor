from bs4 import BeautifulSoup
import requests
import os
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from predictor import inference
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join('static', 'imgs')#'./images/test'
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    preds = ''
    heights = 0.0
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            outputs, preds, heights = inference(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            print(outputs)
            img = os.path.join(app.config['UPLOAD_FOLDER'], filename)#app.config['UPLOAD_FOLDER']

            return render_template('classifier.html', result = '大' if preds=='big' else '小', height = heights, user_image = img)
    return render_template('classifier.html', result = preds, height = heights, user_image = 'static/imgs/cup.png')


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)