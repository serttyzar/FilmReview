from flask import Flask, request, jsonify
import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

app = Flask(__name__)

model = joblib.load('models/logistic_model.pkl')
vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
sia = joblib.load('models/sentiment_analyzer.pkl')

@app.route('/rating', methods=['POST'])
def rate_rewiew():
    data = request.json
    review = data.get("review", "")
    
    review_vectorized = vectorizer.transform([review]).toarray()
    sentiment_scores = sia.polarity_scores(review)
    neg_score = sentiment_scores['neg']
    pos_score = sentiment_scores['pos']
    review_features = np.array([[neg_score, pos_score]])
    review_combined = np.hstack((review_vectorized, review_features)) 
    predicted_proba = model.predict_proba(review_combined)[0]
    predicted_sentiment = model.predict(review_combined)[0]

    rating = int(predicted_proba[1] * 9) + 1
    status = 'Positive' if predicted_sentiment == 1 else 'Negative'

    return jsonify({
        'rating': rating,
        'status': status,
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)