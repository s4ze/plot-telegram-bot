from dotenv import load_dotenv
from os import getenv
from telegram import Update
from telegram.ext import (
    Updater,
    ApplicationBuilder,
    # CommandHandler,
    # MessageHandler,
    filters,
)
from operations import clean_text, fix_spelling, sentiment_dict_load_and_parse

from natasha import (
    MorphVocab,
    Doc,
    NewsEmbedding,
    NewsMorphTagger,
    NewsNERTagger,
    Segmenter,
)
from os import path
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from train_intents import models_dir, vectorizer_file_name, classifier_file_name
import traceback

try:
    vectorizer = load(path.join(models_dir, vectorizer_file_name))
    classifier = load(path.join(models_dir, classifier_file_name))
except FileNotFoundError as e:
    print(f"Не найден файл модели {e}\n{traceback.format_exc()}")


def lemmatize_text(text: str) -> str:
    "Лемматизация (приведение в обычную форму) слов"

    # Обработка текста
    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_morph(morph_tagger)  # Морфологический разбор

    # Извлечение лемм
    if doc.tokens:
        lemmas = []
        for token in doc.tokens:
            token.lemmatize(morph_vocab)
            lemmas.append(token.lemma)

        return " ".join(lemmas)
    else:
        return text


def extract_entities(text: str) -> dict:
    "Извлечение сущностей, данных из текста"

    doc = Doc(text)
    doc.segment(emb)
    doc.tag_morph(morph_vocab)
    doc.tag_ner(ner_tagger)

    entities = {}
    if doc.spans:
        for span in doc.spans:
            entities[span.type] = span.text
    return entities


def analyze_sentiment(text: str) -> str:
    "Анализ тональности слов"
    words = text.split()
    total_score = 0
    word_count = 0

    for word in words:
        # Проверяем, есть ли слово в словаре
        if word in sentiment_dict:
            total_score += sentiment_dict[word]
            word_count += 1

    # Если не найдено ни одного слова, возвращаем нейтральную оценку
    if word_count == 0:
        return "neutral"  # Нейтральная оценка

    # Возвращаем среднюю оценку
    average_score = total_score / word_count
    if average_score > 0.65:
        return "positive"
    elif average_score < 0.35:
        return "negative"
    return "neutral"


def classify_intent(text: str):
    "Классификация намерений"
    vec = vectorizer.transform([text])
    return classifier.predict(vec)[0]


morph_vocab = MorphVocab()
emb = NewsEmbedding()

segmenter = Segmenter()
morph_tagger = NewsMorphTagger(emb)

ner_tagger = NewsNERTagger(emb)

# Загрузка словаря тональности
sentiment_dict = sentiment_dict_load_and_parse("./data/sentiment_dict.txt")


def run_bot():
    load_dotenv()
    token = getenv("TELEGRAM_BOT_API_TOKEN")

    if token is None or token == "":
        print("Ошибка: токен не найден")
        exit(1)

    application = ApplicationBuilder().token(token).build()

    # application.add_handler(CommandHandler("start", start))
    # application.add_handler(CommandHandler("help", help_command))
    # application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, run_bot))

    # Start the Bot
    application.run_polling()
