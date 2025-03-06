from docx import Document
import openai
import random
import json
import nltk
import logging
import tiktoken
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import os
from dotenv import load_dotenv
import fitz
import pytesseract
from pdf2image import convert_from_path
from fastapi.middleware.cors import CORSMiddleware

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()  # Загружаем переменные из .env
openai.api_key = os.getenv("OPENAI_API_KEY")

# Настройка токенизатора
ENCODER = tiktoken.encoding_for_model("gpt-4o-mini")


def count_tokens(text):
    return len(ENCODER.encode(text))


# Фильтр дубликатов
def remove_duplicates(questions):
    unique_questions = []
    seen_questions = set()
    for q in questions:
        if q["question"] not in seen_questions:
            unique_questions.append(q)
            seen_questions.add(q["question"])
    return unique_questions


# Функция для обработки вопросов и ответов из текста, включая теги
def parse_questions(text):
    lines = text.split("\n")
    questions = []
    current_question = None

    for line in lines:
        line = line.strip()  # Убираем лишние пробелы

        if not line:
            continue  # Пропускаем пустые строки

        if line.endswith("?"):  # Если это вопрос
            if current_question:
                questions.append(current_question)
            current_question = {"question": line, "answers": [], "tags": []}
        elif current_question:
            if line.startswith("Теги:"):
                current_question["tags"] = [tag.strip() for tag in line.replace("Теги:", "").split(",")]
            else:
                current_question["answers"].append(line.strip())

    if current_question:
        questions.append(current_question)

    # 🔹 Генерация ответов, если их нет
    for q in questions:
        if not q["answers"]:  # Если у вопроса нет ответа, генерируем его
            correct_answer, wrong_answers = generate_answers(q["question"], q["tags"])
            q["answers"] = [" ".join(correct_answer.strip().splitlines())]  # Объединяем строки правильного ответа
            q["wrong_answers"] = list(set(" ".join(ans.strip().splitlines()) for ans in wrong_answers if ans.strip()))  # Убираем дубли, соединяем строки
            logging.info(f"Сгенерирован ответ: {q['question']} -> {q['answers'][0]} | {q['wrong_answers']}")
        else:
            correct_answer = " ".join(q["answers"]).strip().replace("\n", " ")  # Убираем переносы строк
            wrong_answers = generate_wrong_answers(q["question"], correct_answer, q["tags"])
            q["answers"] = [correct_answer]  # Записываем ответ
            q["wrong_answers"] = list(set(" ".join(ans.strip().splitlines()) for ans in wrong_answers if ans.strip()))  # Убираем дубли, соединяем строки
            logging.info(f"Обработан существующий ответ: {q['question']} -> {q['answers'][0]} | {q['wrong_answers']}")

    print(f"DEBUG (список вопросов после обработки):\n{json.dumps(questions, indent=4, ensure_ascii=False)}")

    return remove_duplicates(questions)



def generate_answers(question, tags):
    try:
        prompt = f"""
Вопрос: {question}
Теги: {', '.join(tags) if tags else 'нет'}
Сгенерируй 1 правильный и 3 правдоподобных, но неверных ответа. 
Ответы должны быть четкими, однострочными.
Формат:
Правильный: [текст правильного ответа]
Неправильные: [ответ 1], [ответ 2], [ответ 3]
        """

        if count_tokens(prompt) > 4096:
            logging.error("Ошибка: текст слишком длинный!")
            return "Ошибка генерации", ["Ошибка генерации 1", "Ошибка генерации 2", "Ошибка генерации 3"]

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты генератор тестов. Придумай 1 правильный и 3 неправильных ответа."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        content = response.choices[0].message.content.strip()
        lines = content.split("\n")

        correct_answer = lines[0].replace("Правильный: ", "").strip()
        wrong_answers = [ans.strip() for ans in lines[1].replace("Неправильные: ", "").split(",")]

        if len(wrong_answers) < 3:
            wrong_answers = ["Ошибка генерации 1", "Ошибка генерации 2", "Ошибка генерации 3"]

        return correct_answer, wrong_answers
    except Exception as e:
        logging.error(f"Ошибка генерации ответов: {e}")
        return "Ошибка генерации", ["Ошибка генерации 1", "Ошибка генерации 2", "Ошибка генерации 3"]


# Генерация неверных ответов с OpenAI API (учет тегов)
def generate_wrong_answers(question, correct_answer, tags):
    try:
        prompt = f"Вопрос: {question}\nПравильный ответ: {correct_answer}\nТеги: {', '.join(tags)}\nСгенерируй 3 неверных ответа, связанных с темой:"

        if count_tokens(prompt) > 4096:
            logging.error("Ошибка: текст слишком длинный!")
            return ["Ошибка генерации 1", "Ошибка генерации 2", "Ошибка генерации 3"]

        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Ты генератор тестов. Создай 3 правдоподобных, но неверных ответа."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )

        wrong_answers = response.choices[0].message.content.strip().split("\n")
        return wrong_answers if len(wrong_answers) >= 3 else ["Ошибка генерации 1", "Ошибка генерации 2",
                                                              "Ошибка генерации 3"]
    except Exception as e:
        logging.error(f"Ошибка генерации неверных ответов: {e}")
        return ["Ошибка генерации 1", "Ошибка генерации 2", "Ошибка генерации 3"]


def read_docx(file_path):
    """Читает текст из .docx файла и возвращает его как строку."""
    doc = Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    return text

def read_pdf(file_path):
    """Читает текст из PDF-файла и возвращает его как строку."""
    doc = fitz.open(file_path)
    text = "\n".join([page.get_text() for page in doc])
    return text


def read_pdf_with_ocr(file_path):
    """Преобразует страницы PDF в изображения и применяет OCR для извлечения текста."""
    images = convert_from_path(file_path)  # Конвертируем PDF в изображения
    text = "\n".join([pytesseract.image_to_string(img, lang="rus+eng") for img in images])  # Распознаем текст

    print(f"OCR текст: {text[:500]}")  # Для отладки

    return text.strip()

# Главная функция обработки тестов
def process_test(file_path, language="ru"):
    try:
        ext = os.path.splitext(file_path)[1].lower()

        if ext == ".docx":
            text = read_docx(file_path)
        elif ext == ".pdf":
            text = read_pdf(file_path)
            if not text.strip():
                logging.warning("PDF содержит изображения, используем OCR")
                text = read_pdf_with_ocr(file_path)
        else:
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()

        if not text.strip():
            raise ValueError("Файл не содержит текста")

        questions = parse_questions(text)

        if not questions:
            logging.error("Ошибка: не удалось извлечь вопросы!")
            raise ValueError("Ошибка обработки вопросов")

        processed_questions = []
        for q in questions:
            if not q["answers"]:  # Если нет ответа, генерируем его с нейросетью
                correct_answer, wrong_answers = generate_answers(q["question"], q["tags"])
            else:  # Если ответ есть, используем его
                correct_answer = q["answers"][0]
                wrong_answers = generate_wrong_answers(q["question"], correct_answer, q["tags"])

            processed_questions.append({
                "question": q["question"],
                "correct_answer": correct_answer,
                "wrong_answers": wrong_answers,
                "tags": q["tags"],
                "language": language
            })

        print(f"DEBUG (готовый JSON):\n{json.dumps(processed_questions, indent=4, ensure_ascii=False)}")
        json_path = "output.json"
        with open(json_path, "w", encoding="utf-8") as json_file:
            json.dump(processed_questions, json_file, indent=4, ensure_ascii=False)

        return processed_questions
    except Exception as e:
        logging.error(f"Ошибка обработки теста: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки теста")


# FastAPI сервер для обработки файлов через API
app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Или укажите ["http://localhost:5176"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/get_questions")
async def get_questions():
    try:
        with open("output.json", "r", encoding="utf-8") as json_file:
            questions = json.load(json_file)
        return {"questions": questions}
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Файл с вопросами не найден")


@app.post("/process_test")
async def upload_file(file: UploadFile = File(...), language: str = "ru"):
    try:
        file_path = f"temp_{file.filename}"
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        results = process_test(file_path, language)
        return {"questions": results}
    except Exception as e:
        logging.error(f"Ошибка обработки запроса: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки запроса")

