import random
from tqdm import tqdm
import pandas as pd
from transformers import pipeline, AutoTokenizer

MODEL_NAME = "slowfastai/DeepSeek-V2-Lite-Chat-bnb-4bit"
OUTPUT_CSV = "../data/datasets/deepseek_generated.csv"
NUM_SAMPLES = 500
BASE_TOPICS = [
    "Любовь", "Деньги", "Здоровье", "Карьера",
    "Путешествия", "Семья", "Дружба",
    "Сложное решение", "Дети", "Еда", "Работа"
]

def main():
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        max_new_tokens=50,
        temperature=0.7,
        do_sample=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    pipe.tokenizer = tokenizer

    def generate_random_topic() -> str:
        """Генерирует случайную тему"""
        prompt = [{
            "role": "user",
            "content": """
                Сгенерируй простую жизненную тему из одного-двух слов.
                Примеры тем: любовь, жизнь, здоровье, деньги, карьера, семья.
                Верни только саму тему, без кавычек.
                Ответь на русском языке.
            """
        }]

        output = pipe(prompt, max_new_tokens=10)
        return output[0]['generated_text'][1]['content']

    def generate_advice(topic: str) -> str:
        """Генерирует предсказание или совет по теме"""
        prompt = [{
            "role": "user",
            "content": f"""Ты — мистический оракул. Дай короткий полушутливый совет
            или предсказание на тему "{topic}".
            Примеры: "Юмор — как зонтик в дождь. Не спасает, но скрашивает.",
            "Песня — это заклинание, которое меняет настроение.",
            "Если кот лёг на клавиатуру — значит, пора отдохнуть.",
            "Чёрная полоса — это просто подготовка к взлёту."
            Ответ начни с 🔮. Ответь на русском языке."""
        }]

        output = pipe(prompt)
        return output[0]['generated_text'][1]['content']

    data = []
    for _ in tqdm(range(NUM_SAMPLES)):
        try:
            if random.random() > 0.5:
                topic = random.choice(BASE_TOPICS)
            else:
                topic = generate_random_topic()
            
            advice = generate_advice(topic)
            data.append({"topic": topic, "advice": advice})
        except Exception as e:
            print(f"Ошибка: {e}")

    df = pd.DataFrame(data)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Данные сохранены в {OUTPUT_CSV}")

if __name__ == "__main__":
    main()