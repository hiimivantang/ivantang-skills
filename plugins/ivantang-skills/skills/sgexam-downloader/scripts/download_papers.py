#!/usr/bin/env python3
"""
SGExam paper downloader — bundled script for the sgexam-downloader skill.
Usage: python download_papers.py --levels 5,6 --subjects maths --types prelim-exam,sa1 --years 2023-2025
"""

import argparse
import random
import re
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE_URL = "https://sgexam.com"
REQUEST_DELAY = (1.0, 2.0)
MAX_PDF_SIZE = 50 * 1024 * 1024

SUBJECTS_BY_LEVEL = {
    1: ["english", "maths", "chinese", "higher-chinese"],
    2: ["english", "maths", "chinese", "higher-chinese"],
    3: ["english", "maths", "science", "chinese", "higher-chinese"],
    4: ["english", "maths", "science", "chinese", "higher-chinese"],
    5: ["english", "maths", "science", "chinese", "higher-chinese"],
    6: ["english", "maths", "science", "chinese", "higher-chinese"],
}

SUBJECT_DISPLAY = {
    "english": "English", "maths": "Maths", "science": "Science",
    "chinese": "Chinese", "higher-chinese": "Higher Chinese",
}

ASSESSMENT_TYPES = [
    "weighted-assessment-1", "weighted-assessment-2", "weighted-assessment-3",
    "non-weighted-assessment-1", "semestral-assessment-1", "semestral-assessment-2",
    "bite-sized-assessment-1", "mid-year-practice-paper", "end-of-year-exam",
    "prelim-exam", "class-test", "revision", "review", "quiz", "term",
    "sa1", "sa2", "ca1", "ca2", "rt",
]
_SORTED_TYPES = sorted(ASSESSMENT_TYPES, key=len, reverse=True)
_SORTED_SUBJECTS = sorted(
    ["higher-chinese", "science", "english", "chinese", "maths"], key=len, reverse=True
)


def get_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        )
    })
    retry = Retry(total=3, backoff_factor=2.0,
                  status_forcelist=[429, 500, 502, 503, 504],
                  allowed_methods=["GET"])
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def fetch(session, url: str) -> BeautifulSoup:
    time.sleep(random.uniform(*REQUEST_DELAY))
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    return BeautifulSoup(resp.text, "lxml")


def extract_paper_urls(soup: BeautifulSoup, level: int, subject: str) -> list:
    slug_re = re.compile(rf"\d{{4}}-p{level}-{re.escape(subject)}-.*?-pdf$")
    seen, urls = set(), []
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("/"): href = BASE_URL + href
        if not href.startswith(BASE_URL): continue
        clean = href.split("?")[0].split("#")[0].rstrip("/")
        slug = clean.rsplit("/", 1)[-1]
        if slug_re.search(slug) and clean not in seen:
            seen.add(clean)
            urls.append(clean + "/")
    return urls


def parse_slug(url: str) -> dict:
    slug = url.rstrip("/").rsplit("/", 1)[-1]
    slug = re.sub(r"-pdf$", "", slug)
    m = re.match(r"^(\d{4})-p(\d)-(.+)$", slug)
    if not m:
        return {"year": None, "level": None, "assessment_type": None, "filename": slug + ".pdf"}
    year, level, remainder = m.group(1), int(m.group(2)), m.group(3)
    for s in _SORTED_SUBJECTS:
        if remainder.startswith(s + "-"):
            remainder = remainder[len(s) + 1:]
            break
    atype = "unknown"
    for t in _SORTED_TYPES:
        if remainder == t or remainder.startswith(t + "-"):
            atype = t
            break
    return {"year": year, "level": level, "assessment_type": atype, "filename": slug + ".pdf"}


def filter_urls(urls: list, years, types) -> list:
    result = []
    for u in urls:
        meta = parse_slug(u)
        if years != "all" and meta["year"] not in years:
            continue
        if types != "all" and meta["assessment_type"] not in types:
            continue
        result.append(u)
    return result


def extract_drive_url(soup: BeautifulSoup):
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "drive.google.com" in href and "export=download" in href:
            return href
        if a.get_text(strip=True).lower() == "download paper":
            return href
    return None


def download_pdf(session, drive_url: str, filepath: Path) -> bool:
    tmp = filepath.with_suffix(".tmp")
    try:
        r = session.get(drive_url, stream=True, timeout=120, allow_redirects=True)
        r.raise_for_status()
        if "text/html" in r.headers.get("Content-Type", ""):
            html = r.text
            soup = BeautifulSoup(html, "lxml")
            confirmed = None
            form = soup.find("form", id="download-form")
            if form and form.get("action"):
                confirmed = form["action"]
            else:
                ca = soup.find("a", href=re.compile(r"confirm="))
                if ca:
                    h = ca["href"]
                    if not h.startswith("http"):
                        h = "https://drive.google.com" + h
                    confirmed = h
                else:
                    mo = re.search(r'confirm=([0-9A-Za-z_-]+)', html)
                    if mo:
                        confirmed = drive_url + f"&confirm={mo.group(1)}"
            if not confirmed:
                return False
            if not (confirmed.startswith("https://drive.google.com") or
                    confirmed.startswith("https://drive.usercontent.google.com")):
                return False
            r = session.get(confirmed, stream=True, timeout=120)
            r.raise_for_status()
        filepath.parent.mkdir(parents=True, exist_ok=True)
        downloaded = 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(8192):
                if chunk:
                    downloaded += len(chunk)
                    if downloaded > MAX_PDF_SIZE:
                        tmp.unlink(missing_ok=True)
                        return False
                    f.write(chunk)
        if tmp.stat().st_size == 0:
            tmp.unlink(missing_ok=True)
            return False
        with open(tmp, "rb") as f:
            if f.read(5) != b"%PDF-":
                tmp.unlink(missing_ok=True)
                return False
        tmp.rename(filepath)
        return True
    except Exception as e:
        print(f"    Error: {e}", file=sys.stderr)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Download SGExam papers")
    p.add_argument("--levels", default="all",
                   help="Comma-separated level numbers (1,2,3) or 'all'")
    p.add_argument("--subjects", default="all",
                   help="Comma-separated subject slugs or 'all'")
    p.add_argument("--types", default="all",
                   help="Comma-separated assessment type slugs or 'all'")
    p.add_argument("--years", default="all",
                   help="Year filter: '2025', '2023,2024', '2020-2025', or 'all'")
    p.add_argument("--output", default="./downloads",
                   help="Output directory (default: ./downloads)")
    return p.parse_args()


def resolve_levels(raw: str) -> list:
    if raw.strip().lower() == "all":
        return list(range(1, 7))
    return [int(x.strip()) for x in raw.split(",")]


def resolve_subjects(raw: str, levels: list) -> list:
    all_subjects = ["english", "maths", "science", "chinese", "higher-chinese"]
    valid = set()
    for lvl in levels:
        valid.update(SUBJECTS_BY_LEVEL[lvl])
    if raw.strip().lower() == "all":
        return [s for s in all_subjects if s in valid]
    selected = [x.strip().lower() for x in raw.split(",")]
    return [s for s in selected if s in valid]


def resolve_years(raw: str):
    if raw.strip().lower() == "all":
        return "all"
    raw = raw.strip()
    if "-" in raw and "," not in raw:
        parts = raw.split("-")
        start, end = int(parts[0]), int(parts[1])
        if start > end:
            start, end = end, start
        return [str(y) for y in range(start, end + 1)]
    return [y.strip() for y in raw.split(",")]


def resolve_types(raw: str):
    if raw.strip().lower() == "all":
        return "all"
    return [t.strip().lower() for t in raw.split(",")]


def main():
    args = parse_args()
    levels = resolve_levels(args.levels)
    subjects = resolve_subjects(args.subjects, levels)
    years = resolve_years(args.years)
    types = resolve_types(args.types)
    output_dir = Path(args.output)

    print("=" * 50)
    print(f"  Levels  : {', '.join(f'P{l}' for l in levels)}")
    print(f"  Subjects: {', '.join(SUBJECT_DISPLAY.get(s, s) for s in subjects)}")
    print(f"  Types   : {types if types == 'all' else ', '.join(types)}")
    print(f"  Years   : {years if years == 'all' else ', '.join(years)}")
    print(f"  Output  : {output_dir}")
    print("=" * 50)

    session = get_session()
    total_dl, total_skip, total_fail = 0, 0, 0
    failures = []

    combos = [(lvl, subj) for lvl in levels for subj in subjects
              if subj in SUBJECTS_BY_LEVEL[lvl]]

    for ci, (level, subject) in enumerate(combos, 1):
        listing_url = f"{BASE_URL}/primary-{level}-{subject}/"
        print(f"\n[{ci}/{len(combos)}] P{level} {SUBJECT_DISPLAY.get(subject, subject)}...")
        try:
            soup = fetch(session, listing_url)
        except Exception as e:
            print(f"  ERROR fetching listing: {e}")
            continue

        all_urls = extract_paper_urls(soup, level, subject)
        filtered = filter_urls(all_urls, years, types)
        print(f"  {len(all_urls)} papers total, {len(filtered)} match filters")

        for pi, paper_url in enumerate(filtered, 1):
            meta = parse_slug(paper_url)
            filename = meta["filename"]
            filepath = output_dir / f"P{level}" / subject / filename
            pfx = f"  [{pi}/{len(filtered)}]"

            if filepath.exists() and filepath.stat().st_size > 0:
                print(f"{pfx} [SKIP] {filename}")
                total_skip += 1
                continue

            try:
                paper_soup = fetch(session, paper_url)
            except Exception as e:
                print(f"{pfx} [FAIL] {filename} — {e}")
                total_fail += 1
                failures.append((filename, str(e)))
                continue

            drive_url = extract_drive_url(paper_soup)
            if not drive_url:
                print(f"{pfx} [FAIL] {filename} — no download link")
                total_fail += 1
                failures.append((filename, "no download link"))
                continue

            time.sleep(random.uniform(*REQUEST_DELAY))
            if download_pdf(session, drive_url, filepath):
                size_kb = filepath.stat().st_size // 1024
                print(f"{pfx} [OK]   {filename} ({size_kb} KB)")
                total_dl += 1
            else:
                print(f"{pfx} [FAIL] {filename} — download failed")
                total_fail += 1
                failures.append((filename, "download failed"))

    print("\n" + "=" * 50)
    print(f"  Downloaded : {total_dl}")
    print(f"  Skipped    : {total_skip}")
    print(f"  Failed     : {total_fail}")
    if failures:
        print("\n  Failed:")
        for name, reason in failures:
            print(f"    - {name}: {reason}")
    print("=" * 50)


if __name__ == "__main__":
    main()
