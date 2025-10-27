from flask import Flask, request, jsonify
app = Flask(__name__)
import pandas as pd
import numpy as np
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import pipeline
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

@app.route('take user input string of car',methods=['POST'])
def search():
    car_name = request.json['car_name']
    params = {'q' : car_name,'restrict_sr' : 'true'}
    res = requests.get('https://oauth.reddit.com/r/CarsIndia/search',headers = headers,params = params)
    #to get commentssss
    p_id = res.json()['data']['children'][0]['data']['id']
    url = f'https://oauth.reddit.com/comments/{p_id}'
    res2 = requests.get(url,headers = headers) 
    comments = res2.json()[1]['data']['children']
    data = []
    for post in res.json()['data']['children']:
        for c in comments:
            data.append({
                'subreddit' : post['data']['subreddit'],
                'title' : post['data']['title'],
                ' selftext' : post['data']['selftext'],
                'comments' : c['data']['body']
            })
    df = pd.DataFrame(data)


    comments = df['comments']
    comments.fillna('none')
    return comments
def analyze_sentiment():
    comments = search()
    result = []
    for i in range(len(comments)):
        val = comments[i]
        print(i ,end = ' ')
        x = sentiment_pipeline(val,truncation = True)
        result.append(int(x[0]['label'][0]))
    final_anus = np.mean(result)
    from collections import Counter
    count = Counter(result)
    return final_anus,count
if __name__ == '__main__':
    app.run(debug=True)

