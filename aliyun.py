# aliyun
from urllib.parse import urlparse
import datetime
import base64
import hmac
import hashlib
import json
import urllib.request

import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO


ak_id = '<your server id>'
ak_secret = '<your server secret>'

def get_current_date():
    date = datetime.datetime.strftime(datetime.datetime.utcnow(), "%a, %d %b %Y %H:%M:%S GMT")
    return date


def to_md5_base64(strBody):
    hash = hashlib.md5()
    hash.update(strBody.encode('utf-8'))
    return base64.b64encode(hash.digest())


def to_sha1_base64(stringToSign, secret):
    hmacsha1 = hmac.new(secret.encode('utf-8'), stringToSign.encode('utf-8'), hashlib.sha1)
    return base64.b64encode(hmacsha1.digest())


def alinet(imgs, clean_tags):
    # imgs: ndarray, (n, c, h, w)
    # scores: ndarray, (n,)
    
    scores = np.zeros(imgs.shape[0])
    imgs = torch.from_numpy(imgs)
    cnt = 0
    
    for img in imgs:
        img = deprocess(img.clone())
        result = alinet_single(img)
        result = json.loads(result)
        s = 0
        for i in result["tags"]:
            if i["value"] in clean_tags:
                s += i["confidence"] / 100
        scores[cnt] = s
        cnt += 1
    
    return scores

def alinet_single(img):
    # img: PIL.Image
    
    buffered = BytesIO()
    img.save(buffered, format="png")
    img64 = base64.b64encode(buffered.getvalue())
    options = {
        'url': 'https://dtplus-cn-shanghai.data.aliyuncs.com/image/tag',
        'method': 'POST',
        'body': json.dumps({"type": 1,
                            "content": img64.decode('utf-8'),
                           }, separators=(',', ':')),
        'headers': {
            'accept': 'application/json',
            'content-type': 'application/json',
            'date':  get_current_date(),
            'authorization': ''
        }
    }

    body = options['body']

    bodymd5 = to_md5_base64(body)

    urlPath = '/image/tag'

    stringToSign = options['method'] + '\n' + options['headers']['accept'] + '\n' + bodymd5.decode('utf-8') + '\n' + options['headers']['content-type'] + '\n' + options['headers']['date'] + '\n' + urlPath
    signature = to_sha1_base64(stringToSign, ak_secret)

    authHeader = 'Dataplus ' + ak_id + ':' + signature.decode('utf-8')
    options['headers']['authorization'] = authHeader

    request = None
    method = options['method']
    url = options['url']

    request = urllib.request.Request(url, body.encode('utf-8'))
    request.get_method = lambda: method
    for key, value in options['headers'].items():
        request.add_header(key, value)


    for cnt_try in range(10):
        try:
            conn = urllib.request.urlopen(request, timeout=5)
            response = conn.read()
            return response.decode('utf-8')

        except Exception as e:
            print(e, "retry", cnt_try)
        
    assert 0
