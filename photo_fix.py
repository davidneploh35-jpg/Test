#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Фотографии каталога -> кадр 3:4 (1920x2560) по правилу своей категории.

Установка (один раз):
    pip install requests beautifulsoup4 pillow numpy mediapipe opencv-python

Три команды:
    # 1) разведка: что вообще есть на сайте, ничего не меняется
    python photo_fix.py https://dr-arushanof.ru/index.php --crawl

    # 2) рабочий прогон: скачать и скадрировать всё, кроме трикотажа
    python photo_fix.py https://dr-arushanof.ru/index.php --crawl --max-pages 300 --fix

    # 3) переобработать уже скачанное, без обращения к сайту
    python photo_fix.py --local photos_ready/original --fix

Правила кадра по категориям задаются таблицей CATEGORY_RULES ниже — правь её под
реальные адреса разделов сайта.
"""

import argparse
import csv
import hashlib
import os
import re
import shutil
import sys
import time
from collections import deque
from io import BytesIO
from urllib.parse import urljoin, urlparse, unquote

from PIL import Image, ImageFilter, ImageOps

Image.MAX_IMAGE_PIXELS = None

# --------------------------------------------------------------------- конфиг

TARGET_W, TARGET_H = 1920, 2560
TARGET_RATIO = TARGET_W / TARGET_H          # 0.75

# Ключевое слово в адресе страницы (или файла) -> правило кадра.
#   full         — в полный рост: от макушки до стоп
#   waist_down   — по пояс: от линии талии до стоп
#   below_waist  — от макушки до точки чуть ниже пояса
#   SKIP         — не обрабатывать вовсе
CATEGORY_RULES = {
    # ключ в адресе   правило кадра    как назвать категорию в имени файла
    "kostyum":       ("full",         "kostyumy"),
    "costume":       ("full",         "kostyumy"),
    "suit":          ("full",         "kostyumy"),

    "bryuk":         ("waist_down",   "bryuki"),
    "bruk":          ("waist_down",   "bryuki"),
    "shtan":         ("waist_down",   "bryuki"),
    "pants":         ("waist_down",   "bryuki"),
    "trousers":      ("waist_down",   "bryuki"),

    "bluz":          ("below_waist",  "bluzki"),
    "blous":         ("below_waist",  "bluzki"),
    "blouse":        ("below_waist",  "bluzki"),
    "rubash":        ("below_waist",  "bluzki"),

    "halat":         ("full",         "halaty"),
    "khalat":        ("full",         "halaty"),
    "gown":          ("full",         "halaty"),
    "robe":          ("full",         "halaty"),

    "trikotazh":     ("SKIP",         "trikotazh"),
    "trikotaj":      ("SKIP",         "trikotazh"),
    "trico":         ("SKIP",         "trikotazh"),
    "knit":          ("SKIP",         "trikotazh"),
}
DEFAULT_RULE = "full"
DEFAULT_CATEGORY = "prochee"

# Отступы, в долях от роста фигуры.
HEADROOM      = 0.06   # запас над макушкой
FOOTROOM      = 0.04   # запас под стопами
WAIST_LIFT    = 0.02   # насколько выше линии бёдер начинать кадр для «по пояс»
BELOW_WAIST   = 0.08   # насколько ниже линии бёдер заканчивать кадр для блузок
SIDE_MARGIN   = 0.08   # запас по бокам от ширины фигуры
PAD_LIMIT     = 0.50   # больше половины кадра фоном не достраиваем: такой кадр негоден

# Доли роста от макушки: где примерно талия и бёдра, если позу найти не удалось.
PROP_HIP = 0.50

MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
             "pose_landmarker_full/float16/1/pose_landmarker_full.task")
MODEL_PATH = os.path.join(os.path.expanduser("~"), ".cache", "photo_fix",
                          "pose_landmarker_full.task")

IMG_EXT = re.compile(r"\.(jpe?g|png|webp|bmp|tiff?)(\?|$)", re.I)
SIZE_SUFFIX = re.compile(r"-\d{2,4}x\d{2,4}(?=\.[A-Za-z]{3,4}$)")
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept": "text/html,image/avif,image/webp,image/*,*/*;q=0.8",
}


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------- сбор ссылок

def extract_from_html(html, base_url):
    """Возвращает (ссылки_на_картинки, ссылки_на_страницы)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    images, pages = set(), set()

    for img in soup.find_all("img"):
        for attr in ("src", "data-src", "data-lazy-src", "data-original", "data-large_image"):
            val = img.get(attr)
            if val:
                images.add(urljoin(base_url, val.strip()))
        if img.get("srcset"):
            for part in img["srcset"].split(","):
                part = part.strip().split()
                if part:
                    images.add(urljoin(base_url, part[0]))

    for src in soup.find_all("source"):
        if src.get("srcset"):
            for part in src["srcset"].split(","):
                part = part.strip().split()
                if part:
                    images.add(urljoin(base_url, part[0]))

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base_url, href)
        if IMG_EXT.search(href):
            images.add(full)          # ссылка на оригинал из галереи
        else:
            pages.add(full)

    # background-image в инлайн-стилях и <style>
    for m in re.finditer(r"url\(\s*(['\"]?)(.*?)\1\s*\)", html, re.I):
        val = m.group(2).strip()
        if IMG_EXT.search(val):
            images.add(urljoin(base_url, val))

    return images, pages


def same_site(url, root_netloc):
    try:
        return urlparse(url).netloc.replace("www.", "") == root_netloc
    except Exception:
        return False


def collect(start_url, crawl=False, max_pages=50, delay=0.3):
    """Возвращает ({адрес_картинки: {страницы_где_найдена}}, [просмотренные_страницы])."""
    import requests

    root_netloc = urlparse(start_url).netloc.replace("www.", "")
    seen_pages, images = set(), {}
    queue = deque([start_url])

    while queue and len(seen_pages) < (max_pages if crawl else 1):
        url = queue.popleft().split("#")[0]
        if url in seen_pages:
            continue
        seen_pages.add(url)
        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            r.raise_for_status()
        except Exception as e:
            log(f"  ! страница не открылась {url}: {e}")
            continue
        ctype = r.headers.get("Content-Type", "").lower()
        looks_html = "<html" in r.text[:2000].lower() or "<body" in r.text[:2000].lower()
        if "html" not in ctype and not looks_html:
            continue
        r.encoding = r.apparent_encoding or r.encoding
        log(f"  страница: {url}")
        imgs, pages = extract_from_html(r.text, url)
        for img in imgs:
            images.setdefault(img, set()).add(url)
        if crawl:
            for p in pages:
                if same_site(p, root_netloc) and p not in seen_pages:
                    queue.append(p)
        time.sleep(delay)

    return images, sorted(seen_pages)


# ------------------------------------------------------------------ категории

def rule_for(candidates, overrides=None):
    """Правило кадра по адресам страниц/файла. Возвращает (правило, категория)."""
    table = dict(CATEGORY_RULES)
    table.update(overrides or {})
    for url in candidates:
        low = url.lower()
        # длинные ключи вперёд, чтобы 'trikotazh' выиграл у более короткого совпадения
        for key in sorted(table, key=len, reverse=True):
            if key in low:
                return table[key]
    return DEFAULT_RULE, DEFAULT_CATEGORY


# ---------------------------------------------------------- скачивание/анализ

def safe_name(url, index):
    name = os.path.basename(unquote(urlparse(url).path)) or f"image_{index}"
    name = re.sub(r"[^\w.\-]", "_", name)
    if not os.path.splitext(name)[1]:
        name += ".jpg"
    return f"{index:03d}_{name}"


def dhash(im, size=8):
    """Разностный хэш: устойчив к пережатию и смене размера."""
    small = im.convert("L").resize((size + 1, size), Image.LANCZOS)
    px = small.tobytes()
    bits = 0
    for row in range(size):
        base = row * (size + 1)
        for col in range(size):
            bits = (bits << 1) | (px[base + col] < px[base + col + 1])
    return bits


def hamming(a, b):
    return bin(a ^ b).count("1")


def open_image(data_or_path):
    im = Image.open(BytesIO(data_or_path) if isinstance(data_or_path, bytes) else data_or_path)
    im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def fetch(url, timeout=60):
    import requests

    r = requests.get(url, headers=HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.content


def download_all(page_map, out_dir, min_side, overrides, delay=0.2):
    """Качает картинки, попутно пробуя найти версию без суффикса размера."""
    os.makedirs(out_dir, exist_ok=True)
    rows, by_md5, tried_full = [], {}, set()

    for i, url in enumerate(sorted(page_map), 1):
        rule, category = rule_for(list(page_map[url]) + [url], overrides)
        if rule == "SKIP":
            rows.append({"url": url, "page": sorted(page_map[url])[0], "file": "",
                         "category": category, "rule": rule, "w": "", "h": "",
                         "status": "skipped_category", "note": "категория исключена"})
            continue

        # полная версия: то же имя без -800x1200
        best_url, best_data = url, None
        bare = SIZE_SUFFIX.sub("", url)
        if bare != url and bare not in tried_full:
            tried_full.add(bare)
            try:
                cand = fetch(bare)
                if len(cand) > 0:
                    best_url, best_data = bare, cand
            except Exception:
                pass

        if best_data is None:
            try:
                best_data = fetch(url)
            except Exception as e:
                rows.append({"url": url, "page": sorted(page_map[url])[0], "file": "",
                             "category": category, "rule": rule, "w": "", "h": "",
                             "status": "download_error", "note": str(e)[:120]})
                continue

        # ключ с категорией: одно и то же фото в двух разделах кадрируется по-разному
        digest = (category, hashlib.md5(best_data).hexdigest())
        if digest in by_md5:
            rows.append({"url": url, "page": sorted(page_map[url])[0], "file": by_md5[digest],
                         "category": category, "rule": rule, "w": "", "h": "",
                         "status": "duplicate", "note": "тот же файл"})
            continue

        try:
            im = open_image(best_data)
            w, h = im.size
        except Exception as e:
            rows.append({"url": url, "page": sorted(page_map[url])[0], "file": "",
                         "category": category, "rule": rule, "w": "", "h": "",
                         "status": "not_an_image", "note": str(e)[:120]})
            continue

        if max(w, h) < min_side:
            rows.append({"url": url, "page": sorted(page_map[url])[0], "file": "",
                         "category": category, "rule": rule, "w": w, "h": h,
                         "status": "too_small", "note": f"иконка/логотип (<{min_side}px)"})
            continue

        fname = safe_name(best_url, i)
        with open(os.path.join(out_dir, fname), "wb") as f:
            f.write(best_data)
        by_md5[digest] = fname

        rows.append({"url": best_url, "page": sorted(page_map[url])[0], "file": fname,
                     "category": category, "rule": rule, "w": w, "h": h,
                     "status": "downloaded",
                     "note": "взят полный оригинал" if best_url != url else ""})
        time.sleep(delay)

    return rows


def drop_near_duplicates(rows, raw_dir, max_distance=3):
    """Из почти одинаковых кадров оставляет самый крупный, остальные помечает.

    Сравниваем только внутри одной категории: у разных изделий кадр и фон похожи,
    и общий список схлопнул бы блузку с брюками.
    """
    kept = {}          # категория -> [(хэш, строка, площадь)]
    for row in rows:
        if row["status"] != "downloaded":
            continue
        try:
            im = open_image(os.path.join(raw_dir, row["file"]))
        except Exception:
            continue
        h = dhash(im)
        area = im.size[0] * im.size[1]
        group = kept.setdefault(row["category"], [])
        twin = None
        for idx, (other_hash, other_row, other_area) in enumerate(group):
            if hamming(h, other_hash) <= max_distance:
                twin = idx
                break
        if twin is None:
            group.append((h, row, area))
            continue
        _, other_row, other_area = group[twin]
        if area > other_area:                       # текущий крупнее — он и остаётся
            other_row["status"] = "duplicate"
            other_row["note"] = f"есть версия крупнее: {row['file']}"
            group[twin] = (h, row, area)
        else:
            row["status"] = "duplicate"
            row["note"] = f"есть версия крупнее: {other_row['file']}"


# ------------------------------------------------------------- поиск фигуры

_LANDMARKER = None
_POSE_STATE = None      # None — не пробовали, "ok" — работает, "off" — недоступна


def ensure_model():
    """Скачивает модель позы один раз в ~/.cache/photo_fix/."""
    if os.path.exists(MODEL_PATH) and os.path.getsize(MODEL_PATH) > 100_000:
        return MODEL_PATH
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    log(f"   качаю модель позы ({os.path.basename(MODEL_PATH)})...")
    data = fetch(MODEL_URL, timeout=180)
    with open(MODEL_PATH, "wb") as f:
        f.write(data)
    return MODEL_PATH


def get_landmarker():
    """Готовит mediapipe. Если его нет — возвращает None, работаем по силуэту."""
    global _LANDMARKER, _POSE_STATE
    if _POSE_STATE == "off":
        return None
    if _LANDMARKER is not None:
        return _LANDMARKER
    try:
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=ensure_model()),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_poses=1,
            output_segmentation_masks=True,
        )
        _LANDMARKER = (mp_vision.PoseLandmarker.create_from_options(options), mp)
        _POSE_STATE = "ok"
        return _LANDMARKER
    except Exception as e:
        _POSE_STATE = "off"
        log(f"   ! поза недоступна ({type(e).__name__}: {str(e)[:90]}), "
            f"работаю по силуэту на фоне")
        return None


def mask_bbox(mask, threshold=0.5, min_fill=0.004):
    """Рамка фигуры по маске. Возвращает (left, top, right, bottom) или None."""
    import numpy as np

    mask = np.asarray(mask)
    if mask.ndim == 3:                   # маска сегментации приходит как (H, W, 1)
        mask = mask[..., 0]
    solid = mask > threshold
    if not solid.any():
        return None
    h, w = solid.shape
    rows = solid.sum(axis=1) > max(1, int(w * min_fill))
    cols = solid.sum(axis=0) > max(1, int(h * min_fill))
    if not rows.any() or not cols.any():
        return None
    ys = np.flatnonzero(rows)
    xs = np.flatnonzero(cols)
    return float(xs[0]), float(ys[0]), float(xs[-1] + 1), float(ys[-1] + 1)


def mask_profile(mask, threshold=0.5, min_run=2):
    """Для каждой строки — левый и правый край фигуры. -1, если строка пустая."""
    import numpy as np

    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask[..., 0]
    solid = mask > threshold
    h, w = solid.shape
    counts = solid.sum(axis=1)
    idx = np.arange(w)[None, :]
    left = np.where(solid, idx, w).min(axis=1).astype(float)
    right = np.where(solid, idx, -1).max(axis=1).astype(float) + 1
    empty = counts < min_run
    left[empty], right[empty] = -1.0, -1.0
    return left, right


def extent_in(body, top, bottom):
    """Ширина фигуры только в полосе кадра: для «по пояс» руки считать не надо."""
    import numpy as np

    profile = body.get("profile")
    if profile is None:
        return body["left"], body["right"]
    row_left, row_right = profile
    lo = max(0, int(top))
    hi = min(len(row_left), int(bottom))
    if hi <= lo:
        return body["left"], body["right"]
    seg_l, seg_r = row_left[lo:hi], row_right[lo:hi]
    valid = seg_l >= 0
    if not valid.any():
        return body["left"], body["right"]
    return float(seg_l[valid].min()), float(seg_r[valid].max())


def pose_body(im):
    """Точки фигуры через mediapipe. None, если человек не найден."""
    pack = get_landmarker()
    if pack is None:
        return None
    landmarker, mp = pack
    import numpy as np

    try:
        arr = np.ascontiguousarray(np.array(im, dtype=np.uint8))
        result = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=arr))
    except Exception as e:
        log(f"   ! детектор позы споткнулся: {type(e).__name__}: {str(e)[:80]}")
        return None
    if not result.pose_landmarks:
        return None

    W, H = im.size
    lm = result.pose_landmarks[0]
    ys = lambda *idx: [lm[i].y * H for i in idx if i < len(lm)]
    xs = lambda *idx: [lm[i].x * W for i in idx if i < len(lm)]

    nose_y = ys(0)[0]
    shoulder_y = sum(ys(11, 12)) / 2
    hip_y = sum(ys(23, 24)) / 2
    foot_pts = ys(27, 28, 29, 30, 31, 32)
    feet_y = max(foot_pts) if foot_pts else hip_y * 1.6

    # макушка: нос лежит примерно на 0.55 высоты головы, плечи — на 1.35
    head_top = nose_y - 0.7 * max(shoulder_y - nose_y, 1.0)

    body_x = xs(11, 12, 23, 24, 25, 26, 27, 28)
    cx = sum(body_x) / len(body_x) if body_x else W / 2
    all_x = [p.x * W for p in lm]
    left, right = min(all_x), max(all_x)

    # силуэт даёт куда более точные макушку и стопы, чем пересчёт по носу
    profile = None
    if getattr(result, "segmentation_masks", None):
        raw_mask = result.segmentation_masks[0].numpy_view()
        box = mask_bbox(raw_mask)
        if box:
            m_left, m_top, m_right, m_bottom = box
            head_top = m_top
            feet_y = max(feet_y, m_bottom)
            left, right = m_left, m_right
            cx = (m_left + m_right) / 2
            profile = mask_profile(raw_mask)

    return {
        "head_top": head_top, "hip": hip_y, "feet": feet_y,
        "left": left, "right": right, "cx": cx, "profile": profile,
        "source": "pose",
        "head_clipped": head_top <= 1.0,
        "feet_clipped": feet_y >= H - 1.0,
    }


def segment_body(im, tolerance=26):
    """Фигура на однотонном фоне: рамка по отличию от цвета фона."""
    import numpy as np

    W, H = im.size
    small = im.resize((max(1, W // 4), max(1, H // 4)), Image.BILINEAR)
    arr = np.asarray(small, dtype=np.int16)
    h, w, _ = arr.shape

    border = np.concatenate([
        arr[:3].reshape(-1, 3), arr[-3:].reshape(-1, 3),
        arr[:, :3].reshape(-1, 3), arr[:, -3:].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    diff = np.abs(arr - bg).sum(axis=2)
    box = mask_bbox(diff.astype(float), threshold=float(tolerance * 3), min_fill=0.02)
    if box is None:
        return None

    scale_x, scale_y = W / w, H / h
    left, top, right, bottom = box
    left, right = left * scale_x, right * scale_x
    top, bottom = top * scale_y, bottom * scale_y
    if (right - left) < W * 0.05 or (bottom - top) < H * 0.15:
        return None                      # похоже на мусор, а не на фигуру
    if (right - left) > W * 0.92 and (bottom - top) > H * 0.92:
        return None                      # фон пёстрый, «фигура» = весь кадр: не верим

    small_l, small_r = mask_profile(diff.astype(float), threshold=float(tolerance * 3))
    rows = np.clip((np.arange(H) / scale_y).astype(int), 0, h - 1)
    row_left, row_right = small_l[rows] * scale_x, small_r[rows] * scale_x
    blank = small_l[rows] < 0
    row_left[blank], row_right[blank] = -1.0, -1.0

    return {
        "head_top": top, "hip": top + PROP_HIP * (bottom - top), "feet": bottom,
        "left": left, "right": right, "cx": (left + right) / 2,
        "profile": (row_left, row_right),
        "source": "silhouette",
        "head_clipped": top <= scale_y,
        "feet_clipped": bottom >= H - scale_y,
    }


def fallback_body(im):
    """Ничего не нашли — считаем, что фигура занимает кадр целиком."""
    W, H = im.size
    return {
        "head_top": 0.0, "hip": PROP_HIP * H, "feet": float(H),
        "left": 0.0, "right": float(W), "cx": W / 2, "profile": None,
        "source": "fallback", "head_clipped": False, "feet_clipped": False,
    }


def find_body(im, use_pose=True):
    if use_pose:
        body = pose_body(im)
        if body:
            return body
    body = segment_body(im)
    return body or fallback_body(im)


# ----------------------------------------------------------------- кадр 3:4

def frame_box(body, rule, W, H):
    """Окно кадра 3:4 по правилу категории. Может выходить за границы — тогда фон."""
    head, hip, feet = body["head_top"], body["hip"], body["feet"]
    span = max(feet - head, 1.0)

    if rule == "waist_down":
        keep_top, keep_bottom = hip - WAIST_LIFT * span, feet + FOOTROOM * span
        protect_head = False
    elif rule == "below_waist":
        keep_top, keep_bottom = head - HEADROOM * span, hip + BELOW_WAIST * span
        protect_head = True
    else:                                    # full
        keep_top, keep_bottom = head - HEADROOM * span, feet + FOOTROOM * span
        protect_head = True

    zone_h = max(keep_bottom - keep_top, 1.0)
    zone_left, zone_right = extent_in(body, keep_top, keep_bottom)
    body_w = max(zone_right - zone_left, 1.0)
    zone_cx = (zone_left + zone_right) / 2

    # окно 3:4: высота по правилу, но фигура должна влезать по ширине с запасом
    width = max(zone_h * TARGET_RATIO, body_w * (1 + 2 * SIDE_MARGIN))
    height = width / TARGET_RATIO

    # достройка фона — крайняя мера: сперва жертвуем боковым запасом, лишь бы влезть
    if height > H or width > W:
        fit_h = min(float(H), W / TARGET_RATIO)
        fit_w = fit_h * TARGET_RATIO
        if fit_w >= body_w and fit_h >= zone_h:
            width, height = fit_w, fit_h

    # по вертикали: голову не режем, лишнее место делим с уклоном вниз
    extra = height - zone_h
    if extra >= 0:
        top = keep_top if rule == "waist_down" else keep_top - extra * 0.35
    else:
        top = keep_top                       # окно ниже зоны — держимся за верх зоны
    bottom = top + height

    # окно влезает по высоте — сдвигаем внутрь, а не режем фоном.
    # для «по пояс» вверх не двигаем: иначе в кадр полезет то, что должно быть срезано
    if height <= H:
        if top < 0:
            top, bottom = 0.0, height
        elif bottom > H and rule != "waist_down":
            top, bottom = H - height, float(H)

    # предохранитель: если фона выходит больше половины кадра, правилом жертвуем —
    # такой кадр всё равно брак. В отчёте доля фона видна, можно посмотреть глазами.
    relaxed = False
    if (max(0.0, -top) + max(0.0, bottom - H)) / height > PAD_LIMIT:
        relaxed = True
        if height <= H:
            top = min(max(top, 0.0), H - height)
        else:
            height, top = float(H), 0.0
            width = height * TARGET_RATIO
        bottom = top + height

    left, right = zone_cx - width / 2, zone_cx + width / 2
    if width <= W:
        if left < 0:
            left, right = 0.0, width
        elif right > W:
            left, right = W - width, float(W)

    return left, top, right, bottom, relaxed


def render(im, box, quality=95):
    """Кроп с достройкой фона по краям + ресайз в 1920x2560."""
    import numpy as np

    W, H = im.size
    left, top, right, bottom = (int(round(v)) for v in box)

    src = (max(left, 0), max(top, 0), min(right, W), min(bottom, H))
    pad_l, pad_t = max(0, -left), max(0, -top)
    pad_r, pad_b = max(0, right - W), max(0, bottom - H)

    box_area = max((right - left) * (bottom - top), 1)
    pad_share = 1.0 - (src[2] - src[0]) * (src[3] - src[1]) / box_area

    crop = im.crop(src)
    if not (pad_l or pad_t or pad_r or pad_b):
        out = crop
    else:
        arr = np.pad(np.asarray(crop, dtype=np.uint8),
                     ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), mode="edge")
        canvas = Image.fromarray(arr)
        radius = max(6, int(max(pad_l + pad_r, pad_t + pad_b) * 0.06))
        out = canvas.filter(ImageFilter.GaussianBlur(radius))
        out.paste(crop, (pad_l, pad_t))     # настоящие пиксели обратно, поверх фона

    upscaled = out.size[0] < TARGET_W
    out = out.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    return out, max(0.0, pad_share), upscaled


# ------------------------------------------------------------------ обработка

REPORT_FIELDS = ["url", "page", "file", "category", "rule", "w", "h",
                 "status", "out_file", "body", "padded", "upscaled", "note"]


def flat_stem(row):
    """Плоское имя: в --local файл может лежать в подпапке, в имя это тащить нельзя."""
    stem = os.path.splitext(os.path.basename(row["file"]))[0]
    return f'{row["category"]}__{stem}'


def out_name(row):
    return flat_stem(row) + ".jpg"


def process_rows(rows, raw_dir, out_dir, args):
    """Кадрирует всё скачанное. Всё готовое — в одну папку out_dir."""
    manual_dir = os.path.join(out_dir, "need_manual")
    os.makedirs(out_dir, exist_ok=True)

    done = manual = skipped = 0
    for row in rows:
        if row["status"] != "downloaded":
            continue
        src = os.path.join(raw_dir, row["file"])
        dst = os.path.join(out_dir, out_name(row))

        if os.path.exists(dst) and not args.redo:
            row["status"] = "skipped_existing"
            row["out_file"] = os.path.basename(dst)
            row["note"] = "уже готово, не переделывал"
            skipped += 1
            continue

        try:
            im = open_image(src)
        except Exception as e:
            row["status"] = "open_error"
            row["note"] = str(e)[:120]
            continue

        body = find_body(im, use_pose=not args.no_pose)
        row["body"] = body["source"]

        # голову режет сам исходник — дорисовать нечем, откладываем
        if body["head_clipped"] and row["rule"] in ("full", "below_waist"):
            os.makedirs(manual_dir, exist_ok=True)
            manual_name = flat_stem(row) + os.path.splitext(row["file"])[1]
            shutil.copy2(src, os.path.join(manual_dir, manual_name))
            row["status"] = "head_cropped_in_source"
            row["out_file"] = f"need_manual/{manual_name}"
            row["note"] = "голова обрезана в оригинале — нужна ручная дорисовка"
            manual += 1
            continue

        left, top, right, bottom, relaxed = frame_box(body, row["rule"], *im.size)
        try:
            out, pad_share, upscaled = render(im, (left, top, right, bottom), args.quality)
            out.save(dst, "JPEG", quality=args.quality, subsampling=0, optimize=True)
        except Exception as e:
            row["status"] = "render_error"
            row["note"] = str(e)[:120]
            continue

        row["status"] = "ready"
        row["out_file"] = os.path.basename(dst)
        row["padded"] = f"{pad_share * 100:.0f}%" if pad_share > 0.005 else ""
        if relaxed:
            row["note"] = "правило кадра ослаблено: иначе фон занял бы полкадра"
        elif pad_share > 0.30:
            row["note"] = "много достроенного фона — посмотреть глазами"
        row["upscaled"] = "да" if upscaled else ""
        done += 1

    return done, manual, skipped


def local_rows(folder, overrides):
    """Режим --local: берём фото из папки, категорию — из пути к файлу."""
    rows = []
    for root, _, files in sorted(os.walk(folder)):
        for i, name in enumerate(sorted(files), 1):
            if not IMG_EXT.search(name):
                continue
            path = os.path.join(root, name)
            rel = os.path.relpath(path, folder)
            rule, category = rule_for([rel.replace(os.sep, "/")], overrides)
            row = {k: "" for k in REPORT_FIELDS}
            row.update({"url": "", "page": rel, "file": os.path.relpath(path, folder),
                        "category": category, "rule": rule})
            if rule == "SKIP":
                row["status"] = "skipped_category"
                row["note"] = "категория исключена"
                rows.append(row)
                continue
            try:
                with Image.open(path) as im:
                    row["w"], row["h"] = im.size
                row["status"] = "downloaded"
            except Exception as e:
                row["status"] = "not_an_image"
                row["note"] = str(e)[:120]
            rows.append(row)
    return rows


def parse_overrides(values):
    out = {}
    for item in values or []:
        if "=" not in item:
            raise SystemExit(f"--rule-override ждёт вид ключ=правило, получено: {item}")
        key, rule = item.split("=", 1)
        if rule not in ("full", "waist_down", "below_waist", "SKIP"):
            raise SystemExit(f"неизвестное правило: {rule}")
        key = key.strip().lower()
        out[key] = (rule, key)
    return out


def summarize(rows):
    titles = {
        "ready": "Готово (1920x2560)",
        "head_cropped_in_source": "Голова обрезана в оригинале -> need_manual",
        "skipped_existing": "Пропущено, уже было готово",
        "skipped_category": "Пропущено по категории (трикотаж)",
        "downloaded": "Скачано, но не обработано (нет --fix)",
        "duplicate": "Дубликаты",
        "too_small": "Мелкие (иконки/логотипы)",
        "download_error": "Не скачались",
        "not_an_image": "Не изображения",
        "open_error": "Не открылись",
        "render_error": "Ошибка обработки",
    }
    groups = {}
    for r in rows:
        groups.setdefault(r["status"], []).append(r)

    log("\n--- ИТОГ ---")
    for key, title in titles.items():
        items = groups.get(key, [])
        if not items:
            continue
        log(f"\n{title}: {len(items)}")
        for r in items[:12]:
            size = f'{r["w"]}x{r["h"]}' if r["w"] else ""
            log(f'   {r["category"]:<12} {size:<12} {r["file"] or r["url"]}')
        if len(items) > 12:
            log(f"   ... ещё {len(items) - 12}")

    by_cat = {}
    for r in rows:
        if r["status"] == "ready":
            by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    if by_cat:
        log("\nПо категориям:")
        for cat, n in sorted(by_cat.items(), key=lambda kv: -kv[1]):
            log(f"   {cat:<14} {n}")


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(
        description="Фото каталога -> 3:4 1920x2560 по правилу категории")
    ap.add_argument("url", nargs="?", help="адрес страницы сайта")
    ap.add_argument("--local", metavar="ПАПКА",
                    help="работать с готовой папкой фото, не трогая сайт")
    ap.add_argument("--out", default="photos_ready", help="папка результата")
    ap.add_argument("--crawl", action="store_true", help="обойти весь сайт, а не одну страницу")
    ap.add_argument("--max-pages", type=int, default=50, help="лимит страниц при --crawl")
    ap.add_argument("--fix", action="store_true", help="кадрировать (без ключа — только отчёт)")
    ap.add_argument("--redo", action="store_true",
                    help="переделать даже то, что уже лежит в папке результата")
    ap.add_argument("--no-pose", action="store_true",
                    help="не использовать mediapipe, только силуэт на фоне")
    ap.add_argument("--min-side", type=int, default=400,
                    help="игнорировать картинки мельче этого (иконки)")
    ap.add_argument("--quality", type=int, default=95, help="качество JPEG")
    ap.add_argument("--rule-override", action="append", metavar="КЛЮЧ=ПРАВИЛО",
                    help="правило для ключевого слова, например bluzki=below_waist")
    args = ap.parse_args()

    if not args.url and not args.local:
        ap.error("нужен либо адрес сайта, либо --local ПАПКА")

    overrides = parse_overrides(args.rule_override)
    os.makedirs(args.out, exist_ok=True)

    if args.local:
        raw_dir = args.local
        log(f"1. Читаю папку {raw_dir} ...")
        rows = local_rows(raw_dir, overrides)
        log(f"   фотографий: {sum(1 for r in rows if r['status'] == 'downloaded')}")
    else:
        raw_dir = os.path.join(args.out, "original")
        log("1. Собираю ссылки на изображения...")
        page_map, pages = collect(args.url, crawl=args.crawl, max_pages=args.max_pages)
        log(f"   страниц просмотрено: {len(pages)}, ссылок на картинки: {len(page_map)}")
        if not page_map:
            log("   Ничего не найдено — возможно, картинки подгружаются скриптом.")
            return 1

        log("\n2. Скачиваю, определяю категорию, ищу полные оригиналы...")
        rows = download_all(page_map, raw_dir, args.min_side, overrides)
        for row in rows:
            for field in REPORT_FIELDS:
                row.setdefault(field, "")
        log("   отсеиваю почти одинаковые кадры...")
        drop_near_duplicates(rows, raw_dir)

    if args.fix:
        log("\n3. Кадрирую под 1920x2560...")
        done, manual, skipped = process_rows(rows, raw_dir, args.out, args)
        log(f"   готово: {done}, в need_manual: {manual}, пропущено готовых: {skipped}")

    report = os.path.join(args.out, "report.csv")
    with open(report, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    summarize(rows)
    log(f"\nОтчёт: {report}")
    if args.fix:
        log(f"Готовые фото: {args.out}")
        manual_dir = os.path.join(args.out, "need_manual")
        if os.path.isdir(manual_dir) and os.listdir(manual_dir):
            log(f"Требуют ручной дорисовки головы: {manual_dir}")
    else:
        log("Чтобы скадрировать — та же команда с ключом --fix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
