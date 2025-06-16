#!/usr/bin/env python
# coding: utf-8
import random
import nltk
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from config import bot_config

X_text = []  # ['Хэй', 'хаюхай', 'Хаюшки', ...]
y = []  # ['hello', 'hello', 'hello', ...]

for intent, intent_data in bot_config["intents"].items():
    for example in intent_data["examples"]:
        X_text.append(example)
        y.append(intent)

vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(3, 3))
X = vectorizer.fit_transform(X_text)
clf = LinearSVC()
clf.fit(X, y)

import os
from telegram import Update, Voice
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    CallbackContext,
)
from gtts import gTTS
import speech_recognition as sr
from pydub import AudioSegment


# def voice_handler(update: Update, context: CallbackContext):
#     "Обработка голосовых сообщений"
#     if update.message:
#         voice = update.message.voice  # Получаем объект голосового сообщения
#         if voice:
#             file = voice.get_file()  # Получаем файл голосового сообщения
#             file.download("voice_message.ogg")  # Сохраняем файл на диск

#     # Преобразуем OGG в WAV для распознавания
#     audio = AudioSegment.from_ogg("voice_message.ogg")
#     audio.export("voice_message.wav", format="wav")

#     # Распознавание речи
#     recognizer = sr.Recognizer()
#     with sr.AudioFile("voice_message.wav") as source:
#         audio_data = recognizer.record(source)  # Читаем аудиофайл
#         try:
#             text = recognizer.recognize_google(
#                 audio_data, language="ru-RU"
#             )  # Распознаем текст
#             context.bot.send_message(
#                 chat_id=update.effective_chat.id, text=f"Распознанный текст: {text}"
#             )
#         except sr.UnknownValueError:
#             context.bot.send_message(
#                 chat_id=update.effective_chat.id, text="Не удалось распознать речь."
#             )
#         except sr.RequestError as e:
#             context.bot.send_message(
#                 chat_id=update.effective_chat.id,
#                 text=f"Ошибка сервиса распознавания: {e}",
#             )


# def main():
#     dp = updater.dispatcher

#     dp.add_handler(MessageHandler(Filters.voice, voice_handler))

#     updater.start_polling()
#     updater.idle()


# if __name__ == "__main__":
#     main()


# Очистка фразы
def clear_phrase(phrase):
    phrase = phrase.lower()

    alphabet = "1234567890abcdefghijklmnopqrstuvwxyzабвгдеёжзийклмнопрстуфхцчшщъыьэюя- "
    result = "".join(symbol for symbol in phrase if symbol in alphabet)

    return result.strip()


# Определение намерений
def classify_intent(replica):
    replica = clear_phrase(replica)

    intent = clf.predict(vectorizer.transform([replica]))[0]

    for example in bot_config["intents"][intent]["examples"]:
        example = clear_phrase(example)
        distance = nltk.edit_distance(replica, example)
        if example and distance / len(example) <= 0.5:
            return intent


# Выдача ответ по намерению
def get_answer_by_intent(intent):
    if intent in bot_config["intents"]:
        responses = bot_config["intents"][intent]["responses"]
        if responses:
            return random.choice(responses)


with open("dialogues.txt") as f:
    content = f.read()

dialogues_str = content.split("\n\n")
dialogues = [dialogue_str.split("\n")[:2] for dialogue_str in dialogues_str]

dialogues_filtered = []
questions = set()

for dialogue in dialogues:
    if len(dialogue) != 2:
        continue

    question, answer = dialogue
    question = clear_phrase(question[2:])
    answer = answer[2:]

    if question != "" and question not in questions:
        questions.add(question)
        dialogues_filtered.append([question, answer])

dialogues_structured = {}  #  {'word': [['...word...', 'answer'], ...], ...}

for question, answer in dialogues_filtered:
    words = set(question.split(" "))
    for word in words:
        if word not in dialogues_structured:
            dialogues_structured[word] = []
        dialogues_structured[word].append([question, answer])

dialogues_structured_cut = {}
for word, pairs in dialogues_structured.items():
    pairs.sort(key=lambda pair: len(pair[0]))
    dialogues_structured_cut[word] = pairs[:1000]

# replica -> word1, word2, word3, ... -> dialogues_structured[word1] + dialogues_structured[word2] + ... -> mini_dataset


def generate_answer(replica):
    replica = clear_phrase(replica)
    words = set(replica.split(" "))
    mini_dataset = []
    for word in words:
        if word in dialogues_structured_cut:
            mini_dataset += dialogues_structured_cut[word]

    # TODO убрать повторы из mini_dataset

    answers = []  # [[distance_weighted, question, answer]]

    for question, answer in mini_dataset:
        if abs(len(replica) - len(question)) / len(question) < 0.2:
            distance = nltk.edit_distance(replica, question)
            distance_weighted = distance / len(question)
            if distance_weighted < 0.2:
                answers.append([distance_weighted, question, answer])

    if answers:
        return min(answers, key=lambda three: three[0])[2]


def get_failure_phrase():
    failure_phrases = bot_config["failure_phrases"]
    return random.choice(failure_phrases)


stats = {"intent": 0, "generate": 0, "failure": 0}


def bot(replica):
    # NLU
    intent = classify_intent(replica)

    # Answer generation

    # выбор заготовленной реплики
    if intent:
        answer = get_answer_by_intent(intent)
        if answer:
            stats["intent"] += 1
            return answer

    # вызов генеративной модели
    answer = generate_answer(replica)
    if answer:
        stats["generate"] += 1
        return answer

    # берем заглушку
    stats["failure"] += 1
    return get_failure_phrase()


bot("Сколько времени?")

############### ТЕЛЕГРАММ ###########################

# https://github.com/python-telegram-bot/python-telegram-bot

from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    if update.message:
        await update.message.reply_text("Привет! Пупупу")


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    if update.message:
        await update.message.reply_text("Смотри что я могу: бам, баум")


async def run_bot(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if update.message:
        replica = update.message.text
        answer = bot(replica)  # Убедитесь, что функция bot определена
        await update.message.reply_text(answer)

        print(stats, replica, answer, "", sep="\n")


async def main():
    """Start the bot."""
    import os
    from dotenv import load_dotenv

    load_dotenv()

    token = os.getenv("TELEGRAM_BOT_API_TOKEN")

    if token is None or token == "":
        print("Ошибка: токен не найден")
        exit(1)

    application = ApplicationBuilder().token(token).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, run_bot))

    # Start the Bot
    application.run_polling()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
