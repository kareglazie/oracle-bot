import pandas as pd
import random
import asyncio
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.ext import ContextTypes, CallbackQueryHandler, CommandHandler, filters
from config import DATA_PATH
from consts import GREETINGS, TOPICS

df = pd.read_csv(DATA_PATH)

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    greeting_style = random.choice(list(GREETINGS.keys()))
    greeting = GREETINGS[greeting_style]
    
    keyboard = [
        [InlineKeyboardButton(topic, callback_data=topic)] 
        for topic in TOPICS
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        f"{greeting}\n\nВыбери тему:",
        reply_markup=reply_markup
    )

async def handle_topic(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик выбора темы"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "new":
        return await handle_new_request(update, context)
    
    selected_topic = query.data
    
    if selected_topic == 'Случайное предсказание':
        advice = df.sample(n=1)['advice'].values[0]
    else:
        topic_lower = selected_topic.lower()
        topic_advices = df[df['topic'].str.lower() == topic_lower]
        if not topic_advices.empty:
            advice = topic_advices.sample(n=1)['advice'].values[0]
        else:
            advice = "🔮 Я не нашел мудрости по этой теме. Попробуй выбрать другую."
    
    typing_message = await query.message.reply_text("🌀 Оракул размышляет...")
    await context.bot.send_chat_action(
        chat_id=query.message.chat_id, 
        action='typing'
    )
    await asyncio.sleep(1)
    await context.bot.delete_message(
        chat_id=query.message.chat_id,
        message_id=typing_message.message_id
    )
    
    keyboard = [[InlineKeyboardButton("🔄 Новое предсказание", callback_data="new")]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        f"📜 {selected_topic.upper()}\n\n{advice}",
        reply_markup=reply_markup
    )

async def handle_new_request(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Обработчик запроса нового предсказания"""
    query = update.callback_query
    await query.answer()
    
    keyboard = [
        [InlineKeyboardButton(topic, callback_data=topic)] 
        for topic in TOPICS
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "Выбери тему для нового предсказания:",
        reply_markup=reply_markup
    )

def setup_handlers(application):
    """Настройка обработчиков"""
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CallbackQueryHandler(handle_topic))
    