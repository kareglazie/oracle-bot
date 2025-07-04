import random
import pandas as pd
from tqdm import tqdm
from transformers import pipeline, AutoTokenizer

MODEL_NAME = "slowfastai/DeepSeek-V2-Lite-Chat-bnb-4bit"
INPUT_CSV = "../data/datasets/deepseek_generated.csv"  
OUTPUT_CSV = "../data/datasets/deepseek_augmented.csv"
AUGMENTATION_FACTOR = 3 
SAMPLE_SIZE = 50

def initialize_model():
    """Инициализация модели и токенизатора"""
    pipe = pipeline(
        "text-generation",
        model=MODEL_NAME,
        max_new_tokens=100,
        temperature=0.7,
        do_sample=True,
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    pipe.tokenizer = tokenizer
    return pipe

def augment_paraphrase(pipe, original_advice: str) -> str:
    """Перефразирует совет, сохраняя смысл"""
    prompt = [{
        "role": "user",
        "content": f"""
        Перефразируй следующий совет, сохраняя его смысл и стиль. 
        Совет должен остаться таким же мистическим и полушутливым.
        
        Оригинальный совет: {original_advice}
        
        Верни только перефразированный вариант, без кавычек.
        Начни с 🔮. Ответь на русском языке.
        """
    }]
    
    output = pipe(prompt)
    return output[0]['generated_text'][1]['content']

def augment_alternative_advice(pipe, topic: str) -> str:
    """Генерирует альтернативный совет на ту же тему"""
    prompt = [{
        "role": "user",
        "content": f"""
        Придумай другой совет или предсказание на тему "{topic}".
        Совет должен быть в мистическом и полушутливом стиле.
        
        Верни только совет, без кавычек.
        Начни с 🔮. Ответь на русском языке.
        """
    }]
    
    output = pipe(prompt)
    return output[0]['generated_text'][1]['content']

def augment_with_context(pipe, topic: str, advice: str) -> str:
    """Добавляет контекст к совету"""
    prompt = [{
        "role": "user",
        "content": f"""
        Тема: {topic}
        Исходный совет: {advice}
        
        Добавь к этому совету небольшое пояснение (2-3 предложения), 
        сохраняя мистический и полушутливый стиль.
        
        Верни только дополненный совет, без кавычек.
        Начни с 🔮. Ответь на русском языке.
        """
    }]
    
    output = pipe(prompt)
    return output[0]['generated_text'][1]['content']

def main():
    pipe = initialize_model()
    
    df = pd.read_csv(INPUT_CSV)
    augmented_data = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        topic = row['topic']
        advice = row['advice']
        
        augmented_data.append({
            "topic": topic,
            "advice": advice,
            "augmentation_type": "original"
        })
        
        for i in range(AUGMENTATION_FACTOR):
            try:
                method = random.choice(['paraphrase', 'alternative', 'context'])
                
                if method == 'paraphrase':
                    new_advice = augment_paraphrase(pipe, advice)
                    augmented_data.append({
                        "topic": topic,
                        "advice": new_advice,
                        "augmentation_type": "paraphrase"
                    })
                    
                elif method == 'alternative':
                    new_advice = augment_alternative_advice(pipe, topic)
                    augmented_data.append({
                        "topic": topic,
                        "advice": new_advice,
                        "augmentation_type": "alternative"
                    })
                        
                elif method == 'context':
                    new_advice = augment_with_context(pipe, topic, advice)
                    augmented_data.append({
                        "topic": topic,
                        "advice": new_advice,
                        "augmentation_type": "with_context"
                    })
                    
            except Exception as e:
                print(f"Ошибка при аугментации: {e}")
                continue

    augmented_df = pd.DataFrame(augmented_data)
    augmented_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Данные сохранены в {OUTPUT_CSV}")

if __name__ == "__main__":
    main()