#!/usr/bin/env python3
from flask import Flask, jsonify, request
from classify import classify_image

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
    f = request.files['image']
    result = dict(
        image=f.filename,
        classification=classify_image(f),
    )
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
