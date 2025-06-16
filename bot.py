from telegram import Update
from telegram.ext import Updater, MessageHandler, filters
from operations import (
    clean_text,
    fix_spelling,
    sentiment_dict_load_and_parse,
)

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
import logging


class Bot:
    morph_vocab: MorphVocab
    emb: NewsEmbedding
    segmenter: Segmenter
    morph_tagger: NewsMorphTagger
    ner_tagger: NewsNERTagger

    sentiment_dict: dict[str, float]

    vectorizer: TfidfVectorizer
    classifier: LinearSVC

    logger: logging.Logger

    def __init__(self):
        # Инициализация компонентов для лемматизации и извлечения сущностей (NER)
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()

        self.segmenter = Segmenter()
        self.morph_tagger = NewsMorphTagger(self.emb)

        self.ner_tagger = NewsNERTagger(self.emb)

        # Загрузка словаря тональности
        self.sentiment_dict = sentiment_dict_load_and_parse("./data/sentiment_dict.txt")

        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

        # Загрузка модели классификации намерений
        try:
            self.vectorizer = load(path.join(models_dir, vectorizer_file_name))
            self.classifier = load(path.join(models_dir, classifier_file_name))
        except FileNotFoundError as e:
            print(f"Не найден файл модели {e}\n{traceback.format_exc()}")

    def lemmatize_text(self, text: str) -> str:
        "Лемматизация (приведение в обычную форму) слов"

        # Обработка текста
        doc = Doc(text)
        doc.segment(self.segmenter)
        doc.tag_morph(self.morph_tagger)  # Морфологический разбор

        # Извлечение лемм
        if doc.tokens:
            lemmas = []
            for token in doc.tokens:
                token.lemmatize(self.morph_vocab)
                lemmas.append(token.lemma)

            return " ".join(lemmas)
        else:
            return text

    def extract_entities(self, text: str) -> dict:
        "Извлечение сущностей, данных из текста"

        doc = Doc(text)
        doc.segment(self.emb)
        doc.tag_morph(self.morph_vocab)
        doc.tag_ner(self.ner_tagger)

        entities = {}
        if doc.spans:
            for span in doc.spans:
                entities[span.type] = span.text
        return entities

    def analyze_sentiment(self, text: str) -> str:
        "Анализ тональности слов"
        words = text.split()
        total_score = 0
        word_count = 0

        for word in words:
            # Проверяем, есть ли слово в словаре
            if word in self.sentiment_dict:
                total_score += self.sentiment_dict[word]
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

    def classify_intent(self, text: str):
        "Классификация намерений"
        vec = self.vectorizer.transform([text])
        return self.classifier.predict(vec)[0]

    def generate_response(self, intent: str, context):
        pass
        pass


async def handle_message(update: Update, context):
    if update.message:
        text = update.message.text
        if text:
            cleaned = clean_text(text)
            corrected = fix_spelling(cleaned)
            lemmas = lemmatize_text(corrected)
            entities = extract_entities(lemmas)
            intent = classify_intent(lemmas)
            response = generate_response(intent, entities)
            await update.message.reply_text(response)


updater = Updater("YOUR_TELEGRAM_TOKEN")
updater.dispatcher.add_handler(MessageHandler(filters.Text(), handle_message))
updater.start_polling()
