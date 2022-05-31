from flask import Flask
from flask_cors import CORS
from tool.tts import tts

app = Flask(__name__)
CORS(app, resources=r'/*')


@app.route('/')
def home():
    return "HCI home"


@app.route('/chat')
def get_chat_answer():
    pass


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
