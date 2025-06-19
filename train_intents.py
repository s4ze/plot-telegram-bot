from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from joblib import dump
from os import path
import shutil

models_dir = "./data/intents_model"


def dataset_load_and_parse(file_path) -> list[tuple[str, str]]:
    "Загрузка датасета для обучения"
    data = []

    try:
        with open(file_path, "r", encoding="utf-8") as file:
            lines = file.readlines()

            for i in range(0, len(lines), 2):
                question = lines[i].strip()
                if i + 1 < len(lines):
                    category = lines[i + 1].strip()
                    data.append((question, category))
    except FileNotFoundError:
        print(f"Файл {file_path} не найден")
    return data


if not path.exists(models_dir):
    print(f"{models_dir} directory doesn't exist")
    exit(1)

dataset_path = "./data/intents_dataset.txt"
data = dataset_load_and_parse(dataset_path)

X = [text for text, label in data]
y = [label for text, label in data]

vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

classifier = LinearSVC()
classifier.fit(X_vec, y)

# Сохранение модели

classifier_file_name = "intent_classifier.pkl"
vectorizer_file_name = "vectorizer.pkl"

dump(classifier, classifier_file_name)
dump(vectorizer, vectorizer_file_name)

shutil.move(classifier_file_name, path.join(models_dir, classifier_file_name))
shutil.move(vectorizer_file_name, path.join(models_dir, vectorizer_file_name))
