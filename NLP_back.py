from flask import Flask, request, jsonify
app = Flask(__name__)
import pandas as pd
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
from collections import Counter
model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)

client = 's3VTNg6KxB7e1tP_JykN2g'
secret_key = '7hxghwdLBZfVmppOPmg-v41cPPY7KA'
auth = requests.auth.HTTPBasicAuth(client,secret_key)
with open('pw.txt','r') as f:
    pw = f.read()
data = {
    'grant_type' : 'password',
    'username'   : 'Ok_Pension_3408',
    'password' : pw
}
headers = {'User-Agent': 'sentiment data/0.0.2'}
res = requests.post('https://www.reddit.com/api/v1/access_token',auth = auth,data = data,headers = headers)
token = res.json()['access_token']
headers['Authorization'] = f'bearer {token}'
requests.get('https://oauth.reddit.com/api/v1/me',headers = headers)

def search(car_name):
    params = {'q': car_name, 'restrict_sr': 'true'}
    res = requests.get('https://oauth.reddit.com/r/CarsIndia/search', headers=headers, params=params)

    posts = res.json()['data']['children']
    if not posts:
        return []

    p_id = posts[0]['data']['id']
    url = f'https://oauth.reddit.com/comments/{p_id}'
    res2 = requests.get(url, headers=headers)
    comments = res2.json()[1]['data']['children']

    data = []
    for post in posts:
        for c in comments:
            data.append({
                'comments': c['data'].get('body', 'none')
            })

    df = pd.DataFrame(data)
    return df['comments'].fillna('none').tolist()


@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    car_name = request.json['car_name']
    comments = search(car_name)

    if not comments:
        return jsonify({"message": "No comments found", "car_name": car_name})

    result = []
    for c in comments:
        pred = sentiment_pipeline(c, truncation=True)
        result.append(int(pred[0]['label'][0]))

    avg_score = float(np.mean(result))
    count = dict(Counter(result))

    return jsonify({
        "car_name": car_name,
        "total_comments": len(comments),
        "avg_sentiment": avg_score,
        "count": count
    })

if __name__ == '__main__':
    app.run(debug=True)

