# src/scrape.py
from __future__ import annotations

import re
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup


# ----------------------------
# Text cleaning
# ----------------------------
def _clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")

    # remove junk
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()

    text = soup.get_text("\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ----------------------------
# URL rules for MedlinePlus
# ----------------------------
AZ_INDEX_RE = re.compile(r"/healthtopics_[a-z]\.html$", re.IGNORECASE)

# Anything that is usually not a "topic page"
SKIP_PATH_PARTS = (
    "/about/",
    "/ency/",
    "/genetics/",
    "/laboratory/",
    "/magazine/",
    "/multiplelanguages/",
    "/news/",
    "/all_easytoread.html",
    "/all_healthtopics.html",
    "/all_howto.html",
    "/sitemap",
)

# File types to skip
SKIP_EXT_RE = re.compile(r"\.(pdf|png|jpg|jpeg|gif|svg|webp|zip)$", re.IGNORECASE)


def _canonicalize(u: str) -> str:
    """Remove fragments, normalize, and drop trailing slash."""
    p = urlparse(u)
    p = p._replace(fragment="", query="")  # drop #... and ?...
    canon = urlunparse(p)
    if canon.endswith("/") and len(canon) > len("https://x/"):
        canon = canon[:-1]
    return canon


def _is_same_domain(seed_netloc: str, u: str) -> bool:
    p = urlparse(u)
    return p.scheme in ("http", "https") and p.netloc == seed_netloc


def _is_good_topic_url(seed_netloc: str, u: str) -> bool:
    """
    Keep topic-like pages, skip navigation/index/search pages.
    MedlinePlus topic pages are commonly:
      https://medlineplus.gov/<topic>.html
    """
    u = _canonicalize(u)
    if not _is_same_domain(seed_netloc, u):
        return False

    p = urlparse(u)
    path = (p.path or "").lower()

    # skip non-html / assets
    if SKIP_EXT_RE.search(path):
        return False

    # skip A-Z index pages
    if AZ_INDEX_RE.search(path):
        return False

    # skip known non-topic sections
    if any(part in path for part in SKIP_PATH_PARTS):
        return False

    # must be a simple one-level .html page: "/insomnia.html"
    # (path.count("/") == 1 means only leading slash)
    if path.endswith(".html") and path.count("/") == 1:
        # avoid very short or generic pages that are often indexes
        filename = path.strip("/").replace(".html", "")
        if filename.startswith("healthtopics_"):
            return False
        if filename in {"index"}:
            return False
        return True

    return False


# ----------------------------
# Crawl
# ----------------------------
def crawl_site(
    seed_url: str,
    max_pages: int = 60,
    extra_seeds: list[str] | None = None,
) -> list[dict]:
    """
    Returns: list of {"url":..., "text":...}
    MedlinePlus-tuned crawler: collects topic pages only.
    """
    seed = urlparse(seed_url)
    seed_netloc = seed.netloc

    seen: set[str] = set()
    queue: list[str] = []

    # Put the best candidates first
    def push(u: str):
        cu = _canonicalize(u)
        if cu and cu not in seen and cu not in queue:
            queue.append(cu)

    push(seed_url)

    if extra_seeds:
        for u in extra_seeds:
            push(u)

    pages: list[dict] = []

    session = requests.Session()
    session.headers.update(
        {
            "User-Agent": "MBAN-RAG-Class-Project/1.0 (+https://example.com)",
            "Accept": "text/html,application/xhtml+xml",
        }
    )

    while queue and len(pages) < max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)

        # filter BEFORE request
        if not _is_good_topic_url(seed_netloc, url):
            continue

        try:
            r = session.get(url, timeout=25)
        except requests.RequestException:
            continue

        ctype = r.headers.get("Content-Type", "")
        if r.status_code != 200 or "text/html" not in ctype:
            continue

        text = _clean_text(r.text)

        # Raise minimum a bit so we avoid very thin pages
        if len(text) < 800:
            continue

        pages.append({"url": url, "text": text})

        # Extract more topic links from this page
        soup = BeautifulSoup(r.text, "lxml")
        for a in soup.select("a[href]"):
            nxt = urljoin(url, a.get("href", ""))
            nxt = _canonicalize(nxt)

            # Only add if it's a good topic page
            if _is_good_topic_url(seed_netloc, nxt):
                # Keep queue from exploding
                if len(queue) < max_pages * 6:
                    push(nxt)

    return pages
