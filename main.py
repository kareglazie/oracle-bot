import asyncio
from telegram.ext import Application, CommandHandler
from config import BOT_TOKEN
from bot.handlers import setup_handlers, start

async def post_init(application):
    """Функция для пост-инициализации"""
    await application.bot.set_my_commands([
        ("start", "Начать диалог с оракулом"),
    ])

def main():
    """Запуск бота"""
    application = Application.builder().token(BOT_TOKEN).post_init(post_init).build()
    
    application.add_handler(CommandHandler("start", start))
    setup_handlers(application)
    
    application.run_polling()

if __name__ == "__main__":
    main()