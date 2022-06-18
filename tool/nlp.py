import json
import requests

signs = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '.', '℃']
sign_to_char = {
    '1': '一',
    '2': '二',
    '3': '三',
    '4': '四',
    '5': '五',
    '6': '六',
    '7': '七',
    '8': '八',
    '9': '九',
    '0': '零',
    '.': '点',
    '℃': '摄氏度'
}


def post(url, data=None):
    data = json.dumps(data, ensure_ascii=False)
    data = data.encode(encoding="utf-8")
    r = requests.post(url=url, data=data)
    return r.json()


def nlp(text):
    sender = 'user'
    url = "http://localhost:5005/webhooks/rest/webhook"
    data = {
        "sender": sender,
        "message": text
    }
    ret = []
    res = post(url, data)
    for item in res:
        ret.append(item['text'])
    print(ret)
    return ret


if __name__ == '__main__':
    print(nlp('你好'))
