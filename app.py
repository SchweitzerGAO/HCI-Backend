from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from tool.tts import tts
from tool.nlp import nlp
import json

app = Flask(__name__)
CORS(app, resources=r'/*')


@app.route('/nlp', methods=['POST'])
def nlp_api():
    data = request.get_data()
    data = json.loads(data.decode("utf-8"))
    text = data['text']
    print(text)
    answer = nlp(text)
    tts(answer)
    return jsonify({'code': 200, 'answer': answer})


@app.route('/tts', methods=['POST'])
def tts_api():
    return send_file('./result/result.wav', as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
