import pickle
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

with open('models/logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

with open('models/sentiment_analyzer.pkl', 'rb') as sia_file:
    sia = pickle.load(sia_file)

def index(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        review = request.POST.get('review')
        if review:
            review_vectorized = vectorizer.transform([review]).toarray()
            sentiment_scores = sia.polarity_scores(review)
            neg_score = sentiment_scores['neg']
            pos_score = sentiment_scores['pos']
            review_features = np.array([[neg_score, pos_score]])  # Преобразуем оценки в массив
            review_combined = np.hstack((review_vectorized, review_features)) 
            predicted_proba = model.predict_proba(review_combined)[0]
            predicted_sentiment = model.predict(review_combined)[0]
            rating = int(predicted_proba[1] * 9) + 1
            status = 'Positive' if predicted_sentiment == 1 else 'Negative'

            return JsonResponse({
                'rating': rating,
                'status': status,
            })
    return JsonResponse({'error': 'No review provided'})