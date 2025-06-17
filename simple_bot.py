from dotenv import load_dotenv
from os import getenv
from telegram import Update
from telegram.ext import (
    Updater,
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)
from operations import (
    clean_text,
    sentiment_dict_load_and_parse,
    extract_price,
    extract_size,
    extract_tags,
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
from data.land_plots import land_plots
from random import choice
from collections import deque


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

    entities.update(extract_price(text))
    entities.update(extract_size(text))

    # return entities
    return {k: v for k, v in entities.items() if v not in ["Бот", "Пользователь"]}


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


def process_text(text: str) -> str:
    "Обработка текста: очистка, исправление ошибок, удаление стоп слов и лемматизация"
    text = clean_text(text)
    text = fix_spelling(text)
    text = clean_stop_words(text)
    return lemmatize_text(text)


def search_plots(entities: dict, tags: list) -> list:
    "Поиск участков по извлеченным параметрам"
    filtered = land_plots.copy()

    # Фильтрация по локации
    if "LOC" in entities:
        location = entities["LOC"].lower()
        filtered = [p for p in filtered if location in p["location"].lower()]

    # Фильтрация по цене
    if "PRICE" in entities:
        price = entities["PRICE"]
        filtered = [p for p in filtered if p["price_value"] <= price]

    # Фильтрация по размеру
    if "SIZE" in entities:
        size = entities["SIZE"]
        filtered = [p for p in filtered if p["size_value"] >= size]

    # Фильтрация по тегам
    if tags:
        filtered = [p for p in filtered if any(tag in p["tags"] for tag in tags)]

    return filtered


def format_short_plot_info(plot: dict) -> str:
    "Краткое описание участка для поисковой выдачи"
    return f"""Участок #{plot['id']}\n\
        📍 {plot['location']} | 📏 {plot['size']}\n\
        💰 {plot['price']}\n\
        🏷️ Теги: #{' #'.join(plot['tags'][:3])}\n"""


def format_land_info(plot: dict) -> str:
    "Описание участка"
    return f"""Участок #{plot['id']}\n\
        📍 {plot['location']} | {plot['size']}\n\
        🌱 Почва: {plot['soil']}\n\
        💵 Цена: {plot['price']}\n\
        ✨ Особенности: {', '.join(plot['features'])}\n\
        📝 {plot['description']}\n"""


def generate_response(
    intent: str, entities: dict, sentiment: str, user_text: str
) -> str:
    # Извлечение тегов из запроса
    tags = extract_tags(user_text)  # Новая функция

    # Обработка поискового запроса
    if any(["поиск", "искать", "предложение", "сравнить"]) in entities:
        found_plots = search_plots(entities, tags)

        if not found_plots:
            response = "По вашему запросу участков не найдено. Попробуйте изменить параметры поиска."
        else:
            # Форматируем первые 3 результата
            plots_list = "\n".join([format_short_plot_info(p) for p in found_plots[:3]])
            response = f"🔍 Найдено участков: {len(found_plots)}\n\n{plots_list}\n"
            response += "Для детальной информации укажите ID участка (например: 'Покажи участок 7')"

        return adapt_to_sentiment(response, sentiment)

    # Обработка запроса деталей участка
    if intent == "plot_details" or "участок" in user_text.lower():
        plot_ids = [int(word) for word in user_text.split() if word.isdigit()]
        if plot_ids:
            plot_id = plot_ids[0]
            plot = next((p for p in land_plots if p["id"] == plot_id), None)
            if plot:
                return format_land_info(plot)
            return "Участок с таким ID не найден. Проверьте номер."

    # 1. Выбираем базовый ответ по намерению
    if intent in intents:
        response = choice(intents[intent])
    else:
        response = choice(intents["default"])

    # 2. Персонализация с использованием сущностей
    personalized_response = personalize_response(response, entities)

    # 3. Адаптация под тональность
    return adapt_to_sentiment(personalized_response, sentiment)


def personalize_response(response: str, entities: dict) -> str:
    "Персонализирует ответ, используя извлеченные сущности"
    # Если есть имя - обращаемся по имени
    if "PER" in entities:
        name = entities["PER"]
        response = f"{name}, {response[0].lower()}{response[1:]}"

    # Для запросов о местоположении - уточняем детали
    if "LOC" in entities and any(
        keyword in response for keyword in ["местоположение", "посёлке", "расположены"]
    ):
        location = entities["LOC"]
        response = response.replace("посёлке", f"посёлке {location}")
        response = response.replace("местоположение", f"местоположение ({location})")

    # Для запросов о цене - учитываем бюджет
    if "PRICE" in entities and "цена" in response:
        price = entities["PRICE"]
        response += f" Упомянутый вами бюджет {price} мы учтём!"

    # Для запросов о размере - предлагаем конкретные варианты
    if "SIZE" in entities and "размер" in response:
        size = entities["SIZE"]
        response = response.replace("Какой размер", f"Для размера {size} соток")

    return response


def adapt_to_sentiment(response: str, sentiment: str) -> str:
    """Адаптирует ответ под тональность сообщения"""
    if sentiment == "negative":
        return "Понимаем ваши сомнения. " + response + " Можем что-то уточнить?"
    elif sentiment == "positive":
        return "Рады вашему интересу! " + response
    return response


def get_context(chat_id: int, new_message: str, is_bot: bool = False) -> str:
    "Возвращает контекст чата (последние 5 пар сообщений)"
    if chat_id not in chat_contexts:
        chat_contexts[chat_id] = deque(maxlen=10)

    role = "Бот" if is_bot else "Пользователь"

    chat_contexts[chat_id].append(f"[{role}]: {new_message}")
    return " ".join(chat_contexts[chat_id])


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

# Создание переменной контекста чатов

chat_contexts = {}

# ===================
# ===================


# Обработчик команды /start
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message:
        if update.effective_chat:
            chat_id = update.effective_chat.id
            if chat_id in chat_contexts:
                chat_contexts[chat_id].clear()
        await update.message.reply_text("Привет! Я ваш чат-бот. Задайте вопрос.")


# Обработчик текстовых сообщений
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if update.message and update.effective_chat:
        try:
            text = update.message.text  # Текст от пользователя
            chat_id = update.effective_chat.id  # ID чата с этим пользователем

            if text and chat_id:
                logger.info(f"Пользователь написал: {text}")

                original_text = text

                # Получаем контекст диалога
                text = get_context(chat_id, text, is_bot=False)

                # Обработка текста
                text = process_text(text)

                # Анализ
                intent = classify_intent(text)
                entities = extract_entities(original_text)
                sentiment = analyze_sentiment(text)

                # Генерация ответа с учётом сущностей и тональности
                response = generate_response(intent, entities, sentiment, original_text)

                get_context(chat_id, response, is_bot=True)

                # Отправка ответа пользователю
                await update.message.reply_text(response)
            else:
                logger.info("Пользователь не написал")
        except Exception as e:
            logger.error(f"Ошибка обработки сообщения {e}\n{traceback.format_exc()}")
            await update.message.reply_text(
                "Произошла ошибка при обработке вашего сообщения"
            )


def run_bot():
    load_dotenv()
    token = getenv("TELEGRAM_BOT_API_TOKEN")

    if token is None or token == "":
        logger.error("Ошибка: токен не найден")
        exit(1)

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    # application.add_handler(CommandHandler("help", help_command))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message)
    )

    # Start the Bot
    application.run_polling()


if __name__ == "__main__":
    run_bot()
