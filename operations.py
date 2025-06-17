from re import sub, search


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
    text = sub(r"[^а-яё\s]", "", text)

    return text


def extract_price(text: str) -> dict:
    "Извлечение цены"
    price_match = search(r"(\d+[\s\d]*)\s*(руб|₽|р\.)", text)
    return {"PRICE": price_match.group()} if price_match else {}


def extract_size(text: str) -> dict:
    "Извлечение размеров участка"
    size_match = search(r"(\d+)\s*(соток|сотки|сотка|га|гектар)", text)
    return {"SIZE": size_match.group()} if size_match else {}


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
