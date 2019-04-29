from flask import Flask,render_template, request, url_for, redirect, jsonify
from flask_cors import CORS
from model import get_pred

app = Flask(__name__)

CORS(app)

@app.route('/')
def index():
    return render_template('index.html', pred='')
@app.route('/get',methods = ['POST'])
def msg():
	txt = request.get_json()['message']
	pred = get_pred(txt)
	return jsonify(pred)
if __name__ == '__main__':
    app.run(debug=True)
