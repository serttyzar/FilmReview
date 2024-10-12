import pickle
from django.shortcuts import render
from django.http import JsonResponse
from sklearn.feature_extraction.text import TfidfVectorizer

with open('models/logistic_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def index(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        review = request.POST.get('review')  # Получаем отзыв из POST-запроса
        if review:
            review_vectorized = vectorizer.transform([review])
            predicted_proba = model.predict_proba(review_vectorized)[0]
            predicted_sentiment = model.predict(review_vectorized)[0]
            rating = int(predicted_proba[1] * 9) + 1
            status = 'Positive' if predicted_sentiment == 1 else 'Negative'

            return JsonResponse({
                'rating': rating,
                'status': status,
            })
    return JsonResponse({'error': 'No review provided'})