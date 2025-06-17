from dotenv import load_dotenv
from os import getenv
from telegram import Update
from telegram.ext import (
    Updater,
    ApplicationBuilder,
    # CommandHandler,
    # MessageHandler,
    filters,
    ContextTypes,
)
from operations import (
    clean_text,
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
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker  # pip install pyspellchecker
from data.intents import intents


def clean_stop_words(text: str) -> str:
    "Удаление стоп-слов (лишних слов)"

    # Разбиение текста на слова
    words = word_tokenize(text, language="russian")
    filtered_text = [word for word in words if word not in stop_words]
    return " ".join(filtered_text)


def fix_spelling(text: str) -> str:
    "Коррекция слов с опечатками"

    words = text.split()

    corrected = []
    for word in words:
        # Обработка слова на опечатки
        corrected_word = spell.correction(word) or word
        corrected.append(corrected_word)
    return " ".join(corrected)


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


def generate_response(intent: str, entities: dict, sentiment: str) -> str:
    "Генерация ответа на основе намерения"
    # Пробуем найти русский эквивалент намерения
    translated_intent = None
    for eng_intent, ru_intent in intent_translation.items():
        if ru_intent == intent:
            translated_intent = eng_intent
            break

    # Если нашли перевод - используем его, иначе оригинальное намерение
    target_intent = translated_intent if translated_intent else intent

    # Получаем возможные ответы для этого намерения
    possible_responses = intents.get(target_intent, None)

    if possible_responses:
        # Выбираем случайный ответ из доступных
        from random import choice

        return choice(possible_responses)
    else:
        # Стандартный ответ, если намерение не распознано
        return (
            f"Я понял ваш запрос как '{intent}'. "
            f"Тональность: {sentiment}. Сущности: {entities}"
        )


# ===================
# ===================

# Добавление и настройка логгера
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Добавление модели для проверки орфографии
spell = SpellChecker(language="ru")

# Загрузка модели классификатора намерений

try:
    vectorizer = load(path.join(models_dir, vectorizer_file_name))
    classifier = load(path.join(models_dir, classifier_file_name))
except FileNotFoundError as e:
    logger.error(f"Не найден файл модели {e}\n{traceback.format_exc()}")
    raise

# Загрузка инструментов для лексиммизации и извлечения сущностей (NER)

morph_vocab = MorphVocab()
emb = NewsEmbedding()

segmenter = Segmenter()
morph_tagger = NewsMorphTagger(emb)

ner_tagger = NewsNERTagger(emb)

# Загрузка базы стоп-слов в русском языке
stop_words = set(stopwords.words("russian"))

# Загрузка словаря тональности
sentiment_dict = sentiment_dict_load_and_parse("./data/sentiment_dict.txt")

# ===================
# ===================


# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        await update.message.reply_text("Привет! Я ваш чат-бот. Задайте вопрос.")


# Обработчик текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        try:
            text = update.message.text  # Текст от пользователя
            logger.info(f"Пользователь написал: {text}")

            if text:
                # Обработка текста
                text = clean_stop_words(fix_spelling(clean_text(text)))
                lemmas = lemmatize_text(text)

                # Анализ
                intent = classify_intent(lemmas)
                entities = extract_entities(lemmas)
                sentiment = analyze_sentiment(lemmas)

                # Генерация ответа с учётом сущностей и тональности
                response = generate_response(intent, entities, sentiment)

                await update.message.reply_text(response)
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения {e}\n{traceback.format_exc()}")
            await update.message.reply_text(
                "Произошла ошибка при обработке вашего сообщения"
            )


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
