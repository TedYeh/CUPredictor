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
    bust_dict = {'A':list(range(70,83)), 'B':list(range(83,85)), 'C':list(range(85,88)),
    'D':list(range(88,90)), 'E':list(range(90,93)), 'F':list(range(93,95)), 'G':list(range(95,98))}
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
            outputs, preds, heights, bust = inference(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            for k, v in bust_dict.items():
                print(v,round(float(bust)))
                if round(float(bust)) in v: 
                    bust = round(float(bust))
                    bust_str = k
                    break
            print(outputs, bust)
            img = os.path.join(app.config['UPLOAD_FOLDER'], filename)#app.config['UPLOAD_FOLDER']

            return render_template('classifier.html', result = '大' if preds=='big' else '小', height = heights, user_image = img, bust_str = bust_str, bust = bust)
    return render_template('classifier.html', result = preds, height = heights, user_image = 'static/imgs/cup.png', bust_str = 'A', bust = 0)


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True, port=8787)