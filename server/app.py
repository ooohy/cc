from flask import Flask, request
import Interface
from flask_cors import CORS

app = Flask(__name__)

@app.route('/trainModel', methods=['POST'])
def trainModel():
    """
    re_json :
    score : float
    """
    args = request.json()
    re_json = Interface.train(args)
    return re_json

@app.route('/loadModek', methods=['POST'])
def loadModlel():
    args = request.json()
    re_json = Interface.classify(args)
    return re_json

app.run(host='127.0.0.1',port=5000)
