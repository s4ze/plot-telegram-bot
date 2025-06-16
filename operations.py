from re import sub as re_sub, findall as re_findall
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from spellchecker import SpellChecker  # pip install pyspellchecker


# def process_user_input(text):
#     # 1. Очистка и исправление опечаток
#     cleaned = clean_text(text)  # "привт хтил айфн 50к" → "привет хотел айфон 50к"
#     # 2. Лемматизация
#     lemmas = lemmatize(cleaned)  # "хотел" → "хотеть"
#     # 3. Извлечение сущностей + цены
#     entities = extract_entities_with_price(lemmas)  # {'PROD': 'айфон', 'PRICE': 50000}
#     # 4. Классификация намерений
#     intent = classify_intent(lemmas)  # "покупка"
#     # 5. Генерация ответа с учётом цены
#     response = generate_response(intent, entities)
#     return response


def clean_text(text: str) -> str:
    "Очистка текста"
    # Приведение к нижнему регистру
    text = text.lower()

    # Удаление спецсимволов и цифр
    text = re_sub(r"[^а-яё\s]", "", text)

    return text


def fix_spelling(text: str) -> str:
    "Коррекция слов с опечатками"

    spell = SpellChecker(language="ru")
    words = text.split()

    corrected = []
    for word in words:
        # Обработка слова на опечатки
        corrected_word = spell.correction(word) or word
        corrected.append(corrected_word)
    return " ".join(corrected)


def clean_stop_words(text: str) -> str:
    "Удаление стоп-слов (лишних слов)"
    # Загрузка базы стоп-слов в русском языке
    stop_words = set(stopwords.words("russian"))

    # Разбиение текста на слова
    words = word_tokenize(text, language="russian")
    filtered_text = [word for word in words if word not in stop_words]
    return " ".join(filtered_text)


def extract_price(text: str) -> int | None:
    "Извлечение числовой цены"
    prices = map(int, re_findall(r"(\d+)\s*(тыс|к|руб|р)?", text.lower()))

    return min(prices) if prices else None


def sentiment_dict_load_and_parse(file_path: str) -> dict[str, float]:
    "Загрузка тонального словаря"
    tonal_dict = {}
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                word, score = line.strip().split("\t")
                tonal_dict[word] = float(score)
    except FileNotFoundError:
        print(f"Файл {file_path} не найден")
    return tonal_dict


text = "Привт я изз москыв довай купимм зимлю рядом"
print(f"Исходный: {text}")
text = clean_text(text)
print(f"Очищенный от лишних символов: {text}")
text = fix_spelling(text)
print(f"Исправленный от опечаток: {text}")
text = clean_stop_words(text)
print(f"Очищенный от стоп-слов: {text}")
text = lemmatize_text(text)
print(f"Лемматизированный: {text}")
entities = extract_entities(text)
print(f"Извлеченные сущности: {entities}")
sentiment = analyze_sentiment(text)
print(f"Оценка тональности: {sentiment}")
