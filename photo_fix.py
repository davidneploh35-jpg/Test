#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Обход сайта -> скачивание всех фотографий -> отчёт по размерам -> приведение к 3:4 (1920x2560).

Установка (один раз):
    pip install requests beautifulsoup4 pillow

Примеры:
    # 1) только посмотреть, что есть на сайте (ничего не меняет)
    python photo_fix.py https://dr-arushanof.ru/index.php

    # 2) скачать всё и сразу сделать 1920x2560
    python photo_fix.py https://dr-arushanof.ru/index.php --fix

    # 3) обойти весь сайт, а не одну страницу
    python photo_fix.py https://dr-arushanof.ru/index.php --crawl --max-pages 100 --fix
"""

import argparse
import csv
import hashlib
import os
import re
import sys
import time
from collections import deque
from io import BytesIO
from urllib.parse import urljoin, urlparse, unquote

import requests
from bs4 import BeautifulSoup
from PIL import Image, ImageOps

Image.MAX_IMAGE_PIXELS = None

TARGET_W, TARGET_H = 1920, 2560
TARGET_RATIO = TARGET_W / TARGET_H          # 0.75
RATIO_TOL = 0.01                            # допуск, чтобы 1199x1600 тоже считалось 3:4
IMG_EXT = re.compile(r"\.(jpe?g|png|webp|bmp|tiff?)(\?|$)", re.I)
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
    root_netloc = urlparse(start_url).netloc.replace("www.", "")
    seen_pages, images = set(), set()
    queue = deque([start_url])

    while queue and len(seen_pages) < (max_pages if crawl else 1):
        url = queue.popleft()
        url = url.split("#")[0]
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
        images |= imgs
        if crawl:
            for p in pages:
                if same_site(p, root_netloc) and p not in seen_pages:
                    queue.append(p)
        time.sleep(delay)

    return sorted(images), sorted(seen_pages)


# ---------------------------------------------------------- скачивание/анализ

def safe_name(url, index):
    name = os.path.basename(unquote(urlparse(url).path)) or f"image_{index}"
    name = re.sub(r"[^\w.\-]", "_", name)
    if not os.path.splitext(name)[1]:
        name += ".jpg"
    return f"{index:03d}_{name}"


def classify(w, h):
    if (w, h) == (TARGET_W, TARGET_H):
        return "exact"
    if h and abs(w / h - TARGET_RATIO) <= RATIO_TOL:
        return "ratio_3x4"
    return "other"


def download_all(urls, out_dir, min_side, delay=0.2):
    os.makedirs(out_dir, exist_ok=True)
    rows, hashes = [], {}

    for i, url in enumerate(sorted(urls), 1):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            data = r.content
        except Exception as e:
            rows.append({"url": url, "file": "", "w": "", "h": "",
                         "ratio": "", "class": "download_error", "note": str(e)})
            continue

        digest = hashlib.md5(data).hexdigest()
        if digest in hashes:
            rows.append({"url": url, "file": hashes[digest], "w": "", "h": "",
                         "ratio": "", "class": "duplicate", "note": "тот же файл"})
            continue

        try:
            im = Image.open(BytesIO(data))
            im = ImageOps.exif_transpose(im)   # учесть поворот из EXIF
            w, h = im.size
        except Exception as e:
            rows.append({"url": url, "file": "", "w": "", "h": "",
                         "ratio": "", "class": "not_an_image", "note": str(e)})
            continue

        if max(w, h) < min_side:
            rows.append({"url": url, "file": "", "w": w, "h": h,
                         "ratio": round(w / h, 4), "class": "too_small",
                         "note": f"иконка/логотип (<{min_side}px)"})
            continue

        fname = safe_name(url, i)
        path = os.path.join(out_dir, fname)
        with open(path, "wb") as f:
            f.write(data)
        hashes[digest] = fname

        rows.append({"url": url, "file": fname, "w": w, "h": h,
                     "ratio": round(w / h, 4), "class": classify(w, h), "note": ""})
        time.sleep(delay)

    return rows


# ------------------------------------------------------------------ обработка

def fix_image(src_path, dst_path, quality=95):
    """Центральный кроп до 3:4 + ресайз в 1920x2560. Возвращает (было, стало, апскейл?)."""
    im = Image.open(src_path)
    im = ImageOps.exif_transpose(im)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    w, h = im.size

    # обрезаем лишнее по длинной стороне, чтобы получить ровно 3:4
    if w / h > TARGET_RATIO:                 # слишком широкая -> режем бока
        new_w = int(round(h * TARGET_RATIO))
        left = (w - new_w) // 2
        im = im.crop((left, 0, left + new_w, h))
    elif w / h < TARGET_RATIO:               # слишком высокая -> режем верх/низ
        new_h = int(round(w / TARGET_RATIO))
        top = (h - new_h) // 2
        im = im.crop((0, top, w, top + new_h))

    upscaled = im.size[0] < TARGET_W
    im = im.resize((TARGET_W, TARGET_H), Image.LANCZOS)
    im.save(dst_path, "JPEG", quality=quality, subsampling=0, optimize=True)
    return (w, h), (TARGET_W, TARGET_H), upscaled


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description="Сбор фотографий с сайта и приведение к 3:4 1920x2560")
    ap.add_argument("url", help="адрес страницы сайта")
    ap.add_argument("--out", default="photos", help="папка результата (по умолчанию ./photos)")
    ap.add_argument("--crawl", action="store_true", help="обойти весь сайт, а не одну страницу")
    ap.add_argument("--max-pages", type=int, default=50, help="лимит страниц при --crawl")
    ap.add_argument("--fix", action="store_true", help="сделать 1920x2560 (кроп по центру + ресайз)")
    ap.add_argument("--fix-all", action="store_true",
                    help="обрабатывать вообще все фото, а не только уже близкие к 3:4")
    ap.add_argument("--min-side", type=int, default=400,
                    help="игнорировать картинки мельче этого (иконки), по умолчанию 400")
    ap.add_argument("--quality", type=int, default=95, help="качество JPEG, по умолчанию 95")
    args = ap.parse_args()

    raw_dir = os.path.join(args.out, "original")
    fixed_dir = os.path.join(args.out, "fixed_1920x2560")
    os.makedirs(args.out, exist_ok=True)

    log("1. Собираю ссылки на изображения...")
    urls, pages = collect(args.url, crawl=args.crawl, max_pages=args.max_pages)
    log(f"   страниц просмотрено: {len(pages)}, найдено ссылок на картинки: {len(urls)}")
    if not urls:
        log("   Ничего не найдено. Возможно, картинки подгружаются скриптом — напиши, разберём отдельно.")
        return

    log("\n2. Скачиваю и измеряю...")
    rows = download_all(urls, raw_dir, args.min_side)

    report = os.path.join(args.out, "report.csv")
    with open(report, "w", newline="", encoding="utf-8-sig") as f:
        wcsv = csv.DictWriter(f, fieldnames=["url", "file", "w", "h", "ratio", "class", "note"])
        wcsv.writeheader()
        wcsv.writerows(rows)

    groups = {}
    for r in rows:
        groups.setdefault(r["class"], []).append(r)

    titles = {
        "exact": "Уже ровно 1920x2560",
        "ratio_3x4": "Соотношение 3:4, но другой размер",
        "other": "Другое соотношение",
        "too_small": "Мелкие (иконки/логотипы) — пропущены",
        "duplicate": "Дубликаты",
        "download_error": "Не скачались",
        "not_an_image": "Не изображения",
    }
    log("\n--- ИТОГ ---")
    for key in ["exact", "ratio_3x4", "other", "too_small", "duplicate",
                "download_error", "not_an_image"]:
        items = groups.get(key, [])
        if not items:
            continue
        log(f"\n{titles[key]}: {len(items)}")
        for r in items[:15]:
            size = f'{r["w"]}x{r["h"]}' if r["w"] else r["note"][:40]
            log(f"   {size:<14} {r['file'] or r['url']}")
        if len(items) > 15:
            log(f"   ... ещё {len(items) - 15}")

    log(f"\nПодробный отчёт: {report}")
    log(f"Оригиналы: {raw_dir}")

    if not args.fix:
        log("\nЧтобы привести к 1920x2560, запусти ту же команду с ключом --fix")
        return

    log("\n3. Привожу к 1920x2560...")
    os.makedirs(fixed_dir, exist_ok=True)
    todo = [r for r in rows
            if r["file"] and r["class"] in (["exact", "ratio_3x4", "other"] if args.fix_all
                                            else ["exact", "ratio_3x4"])]
    done = warned = 0
    for r in todo:
        src = os.path.join(raw_dir, r["file"])
        dst = os.path.join(fixed_dir, os.path.splitext(r["file"])[0] + ".jpg")
        try:
            (ow, oh), _, up = fix_image(src, dst, args.quality)
        except Exception as e:
            log(f"   ! {r['file']}: {e}")
            continue
        done += 1
        if up:
            warned += 1
            log(f"   {r['file']}: {ow}x{oh} -> 1920x2560 (растянуто вверх, качество упадёт)")

    log(f"\nГотово: {done} шт. в {fixed_dir}")
    if warned:
        log(f"Внимание: {warned} фото были меньше 1920 по ширине — их пришлось увеличивать.")
    if not args.fix_all:
        log("Обработаны только фото, уже близкие к 3:4. Нужны все подряд — добавь --fix-all")


if __name__ == "__main__":
    sys.exit(main())
