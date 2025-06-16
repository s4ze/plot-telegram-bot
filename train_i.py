from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

# Пример датасета для обучения
data = [
    ("хочу купить телефон", "покупка"),
    ("сколько стоит доставка", "доставка"),
    ("есть скидки", "скидки"),
]

X = [text for text, label in data]
y = [label for text, label in data]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

classifier = LinearSVC()
classifier.fit(X_vec, y)


def classify_intent(text):
    vec = vectorizer.transform([text])
    return classifier.predict(vec)[0]
