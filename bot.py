import asyncio
import logging
import os
import tempfile

import cv2
import yt_dlp
import requests
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.enums import ParseMode

# ==== НАСТРОЙКИ ====
BOT_TOKEN = os.getenv("BOT_TOKEN", "СЮДА_ВСТАВЬ_ТОКЕН_БОТА")
TRACE_MOE_API = "https://api.trace.moe/search?anilistInfo&cutBorders"
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


def extract_frames(video_path: str, tmp_dir: str, count: int = 3) -> list[str]:
    """Вырезает несколько кадров из видео (начало, середина, конец) и сохраняет как jpg."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Берём кадры на 25%, 50%, 75% видео — избегаем чёрных заставок в начале/конце
    positions = [int(total_frames * p) for p in (0.25, 0.5, 0.75)][:count]
    frame_paths = []

    for i, pos in enumerate(positions):
        cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
        ok, frame = cap.read()
        if ok:
            frame_path = os.path.join(tmp_dir, f"frame_{i}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)

    cap.release()
    return frame_paths


def search_anime_by_image(image_path: str) -> list[dict]:
    """Отправляет один кадр на trace.moe и возвращает список найденных совпадений."""
    with open(image_path, "rb") as f:
        response = requests.post(
            TRACE_MOE_API,
            files={"image": f},
            timeout=60,
        )
    response.raise_for_status()
    data = response.json()
    return data.get("result", [])[:MAX_RESULTS]


def search_anime(video_path: str, tmp_dir: str) -> list[dict]:
    """Пробует несколько кадров из видео, возвращает первый уверенный результат."""
    frame_paths = extract_frames(video_path, tmp_dir)
    if not frame_paths:
        return []

    best_results: list[dict] = []
    for frame_path in frame_paths:
        try:
            results = search_anime_by_image(frame_path)
        except requests.exceptions.RequestException as e:
            logger.warning(f"Кадр {frame_path} не обработан: {e}")
            continue

        if results and results[0].get("similarity", 0) >= MIN_SIMILARITY:
            return results  # нашли уверенное совпадение — не тратим лимит на остальные кадры

        if results and not best_results:
            best_results = results  # запоминаем на случай, если ничего лучше не найдётся

    return best_results


SHIKIMORI_API = "https://shikimori.one/api/animes"


def get_russian_title(english_or_romaji: str) -> str | None:
    """Ищет русское название аниме через Shikimori по английскому/ромадзи названию."""
    if not english_or_romaji:
        return None
    try:
        response = requests.get(
            SHIKIMORI_API,
            params={"search": english_or_romaji, "limit": 1},
            headers={"User-Agent": "AniShazamBot"},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        if data and data[0].get("russian"):
            return data[0]["russian"]
    except Exception as e:
        logger.warning(f"Не удалось получить русское название: {e}")
    return None


def extract_titles(result: dict) -> tuple[str, str | None]:
    """Достаёт английское/ромадзи название и пытается найти русское."""
    anilist = result.get("anilist") or {}
    titles = anilist.get("title") if isinstance(anilist, dict) else {}
    titles = titles or {}

    english = titles.get("english")
    romaji = titles.get("romaji")
    native = titles.get("native")

    display_en = english or romaji or native or result.get("filename") or "Неизвестно"
    lookup_name = english or romaji or native

    russian = get_russian_title(lookup_name) if lookup_name else None
    return display_en, russian


def format_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def build_reply(results: list[dict]) -> str:
    if not results:
        return "😕 Не удалось найти совпадений. Попробуй другой кадр или видео."

    lines = ["🔍 <b>Результаты поиска:</b>\n"]
    for i, r in enumerate(results, start=1):
        similarity = r.get("similarity", 0) * 100
        episode = r.get("episode", "—")
        time_from = format_time(r.get("from", 0))

        title_en, title_ru = extract_titles(r)

        marker = "✅" if similarity >= MIN_SIMILARITY * 100 else "⚠️"
        title_block = f"🇬🇧 {title_en}"
        if title_ru and title_ru != title_en:
            title_block += f"\n   🇷🇺 {title_ru}"

        lines.append(
            f"{marker} <b>{i}.</b> {title_block}\n"
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
            results = await loop.run_in_executor(None, search_anime, video_path, tmp_dir)
        except Exception as e:
            logger.error(f"Ошибка поиска: {e}")
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
