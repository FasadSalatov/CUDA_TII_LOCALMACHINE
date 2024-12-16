from transformers import AutoModelForCausalLM, AutoTokenizer
from telegram import __version__ as TG_VER
from telegram.ext import Application, CommandHandler, MessageHandler, filters
import torch

# Проверка версии python-telegram-bot
try:
    from telegram import __version_info__
    if __version_info__ < (20, 0, 0):
        raise RuntimeError("Требуется версия python-telegram-bot 20.0 или выше.")
except ImportError:
    raise RuntimeError("Требуется версия python-telegram-bot 20.0 или выше.")

# Настройка модели
model_name = "tiiuae/falcon-7b-instruct"
print("Загружаем модель...")

# Определяем устройство для работы (CUDA или CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Используется устройство: {device}")

# Загрузка токенайзера и модели
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if device.type == "cuda" else None,
        torch_dtype=torch.float16 if device.type == "cuda" else torch.float32
    ).to(device)
except Exception as e:
    print(f"Ошибка загрузки модели: {e}")
    exit(1)

def generate_response(prompt):
    """Генерация ответа на основе введенного текста."""
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(
            **inputs, max_new_tokens=150, pad_token_id=tokenizer.eos_token_id
        )
        return tokenizer.decode(outputs[0], skip_special_tokens=True)
    except Exception as e:
        return f"Ошибка генерации ответа: {e}"

# Функции для Telegram бота
async def start(update, context):
    """Приветствие при старте."""
    await update.message.reply_text("Привет! Я локальный ChatGPT. Напиши мне что-нибудь.")

async def respond(update, context):
    """Обработка сообщений и генерация ответов."""
    user_message = update.message.text
    await update.message.reply_text("Обрабатываю ваш запрос...")
    try:
        response = generate_response(user_message)
        await update.message.reply_text(response)
    except Exception as e:
        await update.message.reply_text(f"Произошла ошибка: {e}")

# Главная функция для настройки бота
def main():
    TOKEN = "7708475624:AAFO5ettrpcvGTzbTC-B2OH_0HtZFEMk3hg"

    # Настройка Telegram бота
    application = Application.builder().token(TOKEN).build()

    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, respond))

    print("Бот запущен. Ожидаю сообщений...")
    application.run_polling()

# Точка входа
if __name__ == "__main__":
    main()