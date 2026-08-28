import asyncio
import logging
import os
import tempfile

import yt_dlp
import requests
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.enums import ParseMode

# ==== НАСТРОЙКИ ====
BOT_TOKEN = os.getenv("BOT_TOKEN", "СЮДА_ВСТАВЬ_ТОКЕН_БОТА")
TRACE_MOE_API = "https://api.trace.moe/search?cutBorders"
MAX_RESULTS = 3
MIN_SIMILARITY = 0.85  # ниже этого процента результат считаем ненадёжным

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


def download_video(url: str, output_path: str) -> bool:
    """Скачивает видео по ссылке (TikTok и другие поддерживаемые yt-dlp сайты)."""
    ydl_opts = {
        "outtmpl": output_path,
        "quiet": True,
        "no_warnings": True,
        "format": "mp4/best",
        "max_filesize": 50 * 1024 * 1024,  # 50 МБ лимит
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        return os.path.exists(output_path)
    except Exception as e:
        logger.error(f"Ошибка скачивания видео: {e}")
        return False


def search_anime(video_path: str) -> list[dict]:
    """Отправляет видео на trace.moe и возвращает список найденных совпадений."""
    with open(video_path, "rb") as f:
        response = requests.post(
            TRACE_MOE_API,
            files={"image": f},
            timeout=60,
        )
    response.raise_for_status()
    data = response.json()
    return data.get("result", [])[:MAX_RESULTS]


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def build_reply(results: list[dict]) -> str:
    if not results:
        return "😕 Не удалось найти совпадений. Попробуй другой кадр или видео."

    lines = ["🔍 <b>Результаты поиска:</b>\n"]
    for i, r in enumerate(results, start=1):
        similarity = r.get("similarity", 0) * 100
        anime = r.get("filename") or r.get("anime") or "Неизвестно"
        episode = r.get("episode", "—")
        time_from = format_time(r.get("from", 0))

        marker = "✅" if similarity >= MIN_SIMILARITY * 100 else "⚠️"
        lines.append(
            f"{marker} <b>{i}. {anime}</b>\n"
            f"   Эпизод: {episode}\n"
            f"   Момент: {time_from}\n"
            f"   Совпадение: {similarity:.1f}%\n"
        )

    if results[0].get("similarity", 0) < MIN_SIMILARITY:
        lines.append("\n⚠️ Совпадение низкое — результат может быть неточным.")

    return "\n".join(lines)


@dp.message(CommandStart())
async def cmd_start(message: Message):
    await message.answer(
        "👋 Привет! Я ищу аниме по видео из TikTok.\n\n"
        "Просто пришли мне ссылку на видео из TikTok, "
        "и я скажу, из какого аниме взята сцена."
    )


@dp.message(F.text.contains("tiktok.com"))
async def handle_tiktok_link(message: Message):
    url = message.text.strip()
    status_msg = await message.answer("⏳ Скачиваю видео...")

    with tempfile.TemporaryDirectory() as tmp_dir:
        video_path = os.path.join(tmp_dir, "video.mp4")

        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, download_video, url, video_path)

        if not success:
            await status_msg.edit_text(
                "❌ Не удалось скачать видео. Проверь ссылку — "
                "она должна быть публичной и вести на конкретное видео."
            )
            return

        await status_msg.edit_text("🔎 Ищу совпадение в базе аниме...")

        try:
            results = await loop.run_in_executor(None, search_anime, video_path)
        except requests.exceptions.RequestException as e:
            logger.error(f"Ошибка запроса к trace.moe: {e}")
            await status_msg.edit_text(
                "❌ Сервис поиска временно недоступен. Попробуй позже."
            )
            return

        reply_text = build_reply(results)
        await status_msg.edit_text(reply_text, parse_mode=ParseMode.HTML)


@dp.message()
async def handle_other(message: Message):
    await message.answer(
        "Пришли мне ссылку на видео из TikTok (вида https://www.tiktok.com/...), "
        "и я найду аниме, из которого взята сцена."
    )


async def main():
    logger.info("Бот запущен")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
