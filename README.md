🔮 **Oracle Bot — генератор предсказаний**  

- Выдает рандомные предсказания  
- Дает советы по конкретной теме в мистическо-шутливом стиле  

**ДАННЫЕ:**
1. Генерация: создание датасета с предсказаниями на различные темы (любовь, карьера, деньги и др.) с помощью LLM (`slowfastai/DeepSeek-V2-Lite-Chat-bnb-4bit`).  
2. Аугментация: увеличение разнообразия советов через перефразирование, расширение тем и добавление контекста.  

## 📂 **Структура проекта**  
```
oracle-bot/  
├── data/  
│   ├── datasets/  
│   │   └── deepseek_generated.csv  
|   |   └── deepseek_augmented.csv  
|   |   └── joined.csv
|   ├── scripts/  
│       └── generate_data.py       
│       └── augment_data.py  
│       └── join_data.py        
├── bot/  
│   └── handlers.py
└── README.md  
└── main.py  
└── .env  
└── config.py
└── consts.py
└── requirements.txt
```
