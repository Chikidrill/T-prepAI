import docx
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
        if line.endswith("?"):
            if current_question:
                questions.append(current_question)
            current_question = {"question": line, "answers": [], "tags": []}
        elif current_question and line.strip():
            if line.startswith("Теги:"):
                current_question["tags"] = [tag.strip() for tag in line.replace("Теги:", "").split(",")]
            else:
                current_question["answers"].append(line.strip())

    if current_question:
        questions.append(current_question)

    return remove_duplicates(questions)


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


# Главная функция обработки тестов
def process_test(file_path, language="ru"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        questions = parse_questions(text)
        processed_questions = []

        for q in questions:
            correct_answer = q['answers'][0] if q['answers'] else ""
            wrong_answers = generate_wrong_answers(q['question'], correct_answer, q["tags"])

            processed_questions.append({
                "question": q['question'],
                "correct_answer": correct_answer,
                "wrong_answers": wrong_answers,
                "tags": q["tags"],
                "language": language
            })

        return processed_questions
    except Exception as e:
        logging.error(f"Ошибка обработки теста: {e}")
        raise HTTPException(status_code=500, detail="Ошибка обработки теста")


# FastAPI сервер для обработки файлов через API
app = FastAPI()


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


# Тестирование
if __name__ == "__main__":
    file_path = "test_questions.txt"  # Замени на нужный путь
    test_results = process_test(file_path, language="ru")
    print(json.dumps(test_results, indent=4, ensure_ascii=False))
