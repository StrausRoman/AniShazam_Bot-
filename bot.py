import asyncio
import logging
import os
import random
import tempfile
import time

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
MIN_SIMILARITY = 0.90  # trace.moe официально считает надёжным совпадением >90%

# Настройки многораундового анализа видео
FRAMES_PER_ROUND = 4          # сколько кадров берём за один раунд
MAX_ROUNDS = 5                # максимум раундов (итого до 20 кадров)
MIN_FRAME_SIMILARITY = 0.87   # ниже этого % результат кадра не учитываем в голосовании вообще
CONSENSUS_VOTES = 3            # сколько совпадений одного аниме нужно для досрочной остановки
CONSENSUS_SIMILARITY = 0.93    # средняя схожесть для досрочной остановки
REQUEST_DELAY = 1.3            # пауза между запросами к trace.moe (лимит ~15 запросов/мин)

# Обрезка кадра — убираем зоны, где TikTok обычно рисует текст/подписи/юзернейм,
# чтобы наложенный текст не портил сравнение с оригинальным аниме-кадром
CROP_TOP_RATIO = 0.08     # убираем верхние 8% (шапка, иконки)
CROP_BOTTOM_RATIO = 0.20  # убираем нижние 20% (подпись, музыка, юзернейм)

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


def generate_round_percentages(rounds: int, per_round: int) -> list[list[float]]:
    """
    Генерирует проценты для каждого раунда так, чтобы кадры были разбросаны
    по всему видео и не повторялись между раундами (с небольшим случайным сдвигом).
    """
    total = rounds * per_round
    all_percentages = []
    for i in range(total):
        base = (i + 1) / (total + 1)  # равномерно от ~5% до ~95%
        jitter = random.uniform(-0.03, 0.03)
        pct = min(0.95, max(0.05, base + jitter))
        all_percentages.append(pct)

    random.shuffle(all_percentages)  # чтобы порядок раундов был не строго по порядку видео

    result = []
    for r in range(rounds):
        chunk = all_percentages[r * per_round: (r + 1) * per_round]
        result.append(sorted(chunk))  # внутри раунда сортируем для логичности
    return result


def crop_overlay_regions(frame):
    """
    Обрезает верх и низ кадра, где TikTok обычно размещает текст, подписи,
    юзернейм и иконки — это снижает влияние наложенного текста на распознавание.
    """
    height, width = frame.shape[:2]
    top = int(height * CROP_TOP_RATIO)
    bottom = int(height * (1 - CROP_BOTTOM_RATIO))
    if bottom <= top:
        return frame  # на случай совсем маленького видео — не обрезаем
    return frame[top:bottom, :]


def extract_frame_at_percentage(video_path: str, tmp_dir: str, percentage: float, idx: int) -> str | None:
    """Вырезает один кадр на заданном проценте длительности видео и обрезает зоны с текстом."""
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return None

    pos = int(total_frames * percentage)
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        return None

    frame = crop_overlay_regions(frame)

    frame_path = os.path.join(tmp_dir, f"frame_{idx}.jpg")
    cv2.imwrite(frame_path, frame)
    return frame_path


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


def _candidate_key(result: dict) -> str:
    """Ключ для группировки результатов по конкретному аниме (по AniList ID, если есть)."""
    anilist = result.get("anilist")
    if isinstance(anilist, dict) and anilist.get("id"):
        return f"anilist:{anilist['id']}"
    return f"filename:{result.get('filename', 'unknown')}"


def search_anime(video_path: str, tmp_dir: str) -> list[dict]:
    """
    Многораундовый поиск: берёт несколько кадров за раунд на разных процентах видео,
    сверяет результаты между раундами и голосованием выбирает самое частое и уверенное совпадение.
    """
    round_percentages = generate_round_percentages(MAX_ROUNDS, FRAMES_PER_ROUND)
    candidates: dict[str, dict] = {}
    frame_idx = 0

    for round_num, percentages in enumerate(round_percentages, start=1):
        for pct in percentages:
            frame_path = extract_frame_at_percentage(video_path, tmp_dir, pct, frame_idx)
            frame_idx += 1
            if not frame_path:
                continue

            try:
                results = search_anime_by_image(frame_path)
            except requests.exceptions.RequestException as e:
                logger.warning(f"Кадр на {pct*100:.0f}% не обработан: {e}")
                time.sleep(REQUEST_DELAY)
                continue

            time.sleep(REQUEST_DELAY)

            if not results:
                continue

            top = results[0]
            similarity = top.get("similarity", 0)
            if similarity < MIN_FRAME_SIMILARITY:
                continue  # слабое совпадение считаем шумом и не голосуем за него

            key = _candidate_key(top)
            entry = candidates.setdefault(key, {"result": top, "similarities": [], "count": 0})
            entry["count"] += 1
            entry["similarities"].append(similarity)
            if similarity > entry["result"].get("similarity", 0):
                entry["result"] = top  # сохраняем пример с лучшим совпадением для эпизода/тайминга

        # После каждого раунда проверяем, не набрался ли уверенный консенсус
        if candidates:
            leader = max(
                candidates.values(),
                key=lambda e: (e["count"], sum(e["similarities"]) / len(e["similarities"])),
            )
            avg_similarity = sum(leader["similarities"]) / len(leader["similarities"])
            if leader["count"] >= CONSENSUS_VOTES and avg_similarity >= CONSENSUS_SIMILARITY:
                logger.info(f"Консенсус достигнут после раунда {round_num}/{MAX_ROUNDS}")
                break

    if not candidates:
        return []

    ranked = sorted(
        candidates.values(),
        key=lambda e: (e["count"], sum(e["similarities"]) / len(e["similarities"])),
        reverse=True,
    )

    final_results = []
    for entry in ranked[:MAX_RESULTS]:
        r = dict(entry["result"])
        r["similarity"] = sum(entry["similarities"]) / len(entry["similarities"])
        r["votes"] = entry["count"]
        r["frames_checked"] = frame_idx
        final_results.append(r)

    return final_results


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
        return (
            "😕 Не удалось найти совпадений.\n\n"
            "Возможно, в кадрах слишком много текста/наложений, "
            "или сцена взята не из аниме, а из фан-эдита/арта."
        )

    # Если даже лучший результат ниже надёжного порога — честно говорим, что не уверены,
    # а не выдаём случайное совпадение как будто это точный ответ
    if results[0].get("similarity", 0) < MIN_FRAME_SIMILARITY:
        return (
            "😕 Не нашёл уверенного совпадения — лучший вариант был ниже "
            f"{int(MIN_FRAME_SIMILARITY * 100)}%, показывать его не буду, "
            "слишком велика вероятность ошибки.\n\n"
            "Попробуй скинуть видео без текста/фильтров или другой момент сцены."
        )

    total_checked = results[0].get("frames_checked", "?")
    lines = [f"🔍 <b>Результаты поиска</b> (проверено кадров: {total_checked}):\n"]

    for i, r in enumerate(results, start=1):
        similarity = r.get("similarity", 0) * 100
        episode = r.get("episode", "—")
        time_from = format_time(r.get("from", 0))
        votes = r.get("votes", 1)

        title_en, title_ru = extract_titles(r)

        marker = "✅" if similarity >= MIN_SIMILARITY * 100 else "⚠️"
        title_block = f"🇬🇧 {title_en}"
        if title_ru and title_ru != title_en:
            title_block += f"\n   🇷🇺 {title_ru}"

        lines.append(
            f"{marker} <b>{i}.</b> {title_block}\n"
            f"   Эпизод: {episode}\n"
            f"   Момент: {time_from}\n"
            f"   Совпадение: {similarity:.1f}% (подтверждено кадрами: {votes})\n"
        )

    if results[0].get("similarity", 0) < MIN_SIMILARITY:
        lines.append("\n⚠️ Совпадение среднее (87–90%) — стоит перепроверить глазами.")

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

        await status_msg.edit_text(
            "🔎 Анализирую видео по нескольким кадрам и сверяю результаты...\n"
            "Это может занять до минуты."
        )

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
