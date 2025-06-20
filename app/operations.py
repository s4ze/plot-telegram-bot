from re import sub, search, compile


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


# def extract_price(text: str) -> dict:
#     "Извлечение цены"
#     match = search(r"(\d+[\s\d]*)\s*(тыс|руб|₽|р\.)", text)
#     if match:
#         price = int(match.group(1).replace(" ", ""))
#         if "тыс" in text:
#             price *= 1000
#         return {"PRICE": price}
#     return {}


# def extract_size(text: str) -> dict:
#     "Извлечение размеров участка"
#     match = search(r"(\d+)\s*(соток|сотки|сотка|га|гектар)", text)
#     if match:
#         print(f"MATCH::: {int(match.group(1).replace(" ", ""))}")
#         return {"PRICE": int(match.group(1).replace(" ", ""))}
#     return {}


def extract_price(text: str) -> dict[str, int]:
    """
    Извлечение цены.
    Поддерживаются форматы:
      - 500 тыс. руб.
      - 500 тысяч рублей
      - 500000 ₽
      - 1.2 млн руб.
    """
    pattern = compile(
        r"(?P<number>\d+(?:[ \d]*)(?:[.,]\d+)?)\s*"
        r"(?P<unit>тыс\.?|тысяч|млн\.?|миллион|миллионов)?\s*"
        r"(?P<currency>руб(?:\.|лей)?|₽)\b",
    )
    m = pattern.search(text)
    if not m:
        return {}

    num = m.group("number").replace(" ", "").replace(",", ".")
    price = float(num)
    unit = m.group("unit") or ""
    unit = unit.lower()
    # Унифицируем в рубли
    if "млн" in unit or "миллион" in unit:
        price *= 1_000_000
    elif "тыс" in unit or "тысяч" in unit:
        price *= 1_000

    return {"PRICE": int(price)}


def extract_size(text: str) -> dict[str, int]:
    """
    Извлечение площади участка.
    Поддерживаются форматы:
      - 10 соток
      - 0.5 га
      - 1 гектар
    """
    # Регекс на сотки/гектары
    pattern = compile(
        r"(?P<number>\d+(?:[.,]\d+)?)\s*" r"(?P<unit>соток|сотк[аи]|га|гектар)\b",
    )
    m = pattern.search(text)
    if not m:
        return {}

    num = m.group("number").replace(",", ".")
    size = float(num)
    unit = m.group("unit").lower()
    # Переводим в сотки
    if "га" in unit or "гектар" in unit:
        size *= 100  # 1 га = 100 соток

    return {"SIZE": int(size)}


def extract_tags(text: str) -> list:
    """Извлечение тегов из текста (ключевых слов)"""
    keywords = [
        "солнечный",
        "речка" "лесной" "лес",
        "река",
        "озеро",
        "море",
        "гора",
        "город",
        "село",
        "ижс",
        "коммерция",
        "дача",
        "вода",
        "газ",
        "электричество",
        "песок",
        "глина",
    ]
    found = []
    for word in text.lower().split():
        if word in keywords:
            found.append(word)
    return found


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
