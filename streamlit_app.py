#!/usr/bin/env python3
"""
TV/Movie Outfit Matcher — Streamlit App
=======================================
Upload photos of your clothing and discover if (and where) similar items appeared on TV shows or movies.

How it works
------------
1) You upload 1–5 images of a garment.
2) The app uses CLIP (open_clip_torch) to predict fashion tags (color, garment type, pattern, style, possible brands).
3) It converts those tags into smart queries.
4) It searches well‑known "seen on screen" databases (WornOnTV, ShopYourTV, Spotern, TheTake, Filmgarb).
5) You get clickable results with titles, pages, and thumbnails (when available), plus a CSV download.

Install & run
-------------
# fresh env (recommended)
python -m venv .venv  # or: python3 -m venv .venv
# Windows
.\\.venv\\Scripts\\Activate
# macOS/Linux
source .venv/bin/activate

pip install streamlit open_clip_torch torch torchvision pillow beautifulsoup4 requests

# launch
streamlit run image_to_screen_matcher_app.py

Notes
-----
- CLIP can run on CPU; GPU (CUDA/MPS) will be auto‑used if available.
- This app queries public search pages politely (low request rate). Respect each site's Terms.
- Results are *best‑effort matches* — always verify on the linked pages.

"""
from __future__ import annotations
import io
import time
import re
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict

import requests
from bs4 import BeautifulSoup
from PIL import Image
import streamlit as st

# ----------------- CLIP (open_clip) -----------------
try:
    import torch
    import open_clip
except Exception:
    open_clip = None
    torch = None

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
HEADERS = {"User-Agent": USER_AGENT}
TIMEOUT = 20

# ----------------- Fashion vocabulary -----------------
COLORS = [
    "black", "white", "ivory", "cream", "beige", "tan", "brown", "camel", "gray", "charcoal", "silver", "gold",
    "navy", "blue", "light blue", "teal", "green", "olive", "sage", "mint", "yellow", "mustard", "orange", "rust",
    "red", "burgundy", "pink", "hot pink", "fuchsia", "magenta", "purple", "lilac"
]
GARMENTS = [
    "blazer", "tweed blazer", "trench coat", "leather jacket", "denim jacket", "bomber jacket", "cardigan", "sweater",
    "turtleneck", "hoodie", "sweatshirt", "button-up shirt", "oxford shirt", "blouse", "camisole", "bustier",
    "corset top", "t-shirt", "tank top", "bodysuit", "mini dress", "midi dress", "maxi dress", "slip dress",
    "bodycon dress", "wrap dress", "shirt dress", "sheath dress", "fit and flare dress", "gown", "jumpsuit", "romper",
    "jeans", "straight-leg jeans", "skinny jeans", "wide-leg pants", "trousers", "cargo pants", "leggings",
    "mini skirt", "midi skirt", "maxi skirt", "pencil skirt", "pleated skirt", "tennis skirt", "shorts"
]
PATTERNS = [
    "solid", "striped", "plaid", "gingham", "houndstooth", "tweed", "polka dot", "floral", "animal print",
    "leopard print", "snake print", "paisley", "argyle", "sequin", "lace", "satin", "silk", "velvet", "leather",
    "denim", "corduroy", "ribbed", "cable knit", "mesh", "sheer"
]
STYLES = [
    "double-breasted", "gold buttons", "pearl buttons", "embellished", "fringe", "ruffle", "puff sleeve",
    "off-shoulder", "one-shoulder", "halter", "v-neck", "square neckline", "high neck", "asymmetric", "belted", "peplum"
]
BRANDS_COMMON = [
    # Popular mass & designer — zero-shot brand guesses are noisy; treat as hints only
    "Zara", "H&M", "Mango", "Uniqlo", "COS", "Aritzia", "Reformation", "Abercrombie", "J.Crew", "Banana Republic",
    "Anthropologie", "Free People", "Levi's", "Nike", "Adidas", "Lululemon",
    "Burberry", "Chanel", "Gucci", "Prada", "Dior", "Valentino", "Saint Laurent", "Balenciaga", "Miu Miu",
    "Bottega Veneta", "The Row", "Ferragamo", "Fendi", "Versace"
]

# ----------------- Data classes -----------------
@dataclass
class Hit:
    source: str
    title: str
    url: str
    snippet: str
    thumb: str = ""

# ----------------- Streamlit cache helpers -----------------
@st.cache_resource(show_spinner=False)
def load_clip(device: str = "auto", model_name: str = "ViT-B-32", pretrained: str = "openai"):
    if open_clip is None:
        raise RuntimeError("open_clip_torch/torch not installed. pip install open_clip_torch torch torchvision")
    if device == "auto":
        if torch and torch.cuda.is_available():
            device = "cuda"
        elif torch and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval(); model.to(device)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model, preprocess, tokenizer, device

@st.cache_data(show_spinner=False)
def http_get(url: str, params: Optional[dict] = None) -> Optional[str]:
    try:
        r = requests.get(url, params=params, headers=HEADERS, timeout=TIMEOUT)
        if r.status_code == 200:
            return r.text
        return None
    except requests.RequestException:
        return None

@st.cache_data(show_spinner=False)
def fetch_thumb(url: str) -> str:
    """Try to get a representative image (og:image) from a result page."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        if not r.ok:
            return ""
        soup = BeautifulSoup(r.text, "html.parser")
        og = soup.find("meta", {"property": "og:image"})
        return og["content"] if og and og.get("content") else ""
    except Exception:
        return ""

# ----------------- CLIP tagger -----------------
def rank_labels(img: Image.Image, labels: List[str], prefix: str = "a photo of a") -> List[Tuple[str, float]]:
    model, preprocess, tokenizer, device = load_clip()
    with torch.no_grad():
        img_t = preprocess(img).unsqueeze(0).to(device)
        texts = [f"{prefix} {lab}" for lab in labels]
        tok = tokenizer(texts).to(device)
        image_features = model.encode_image(img_t)
        text_features = model.encode_text(tok)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        sims = (100.0 * image_features @ text_features.T).squeeze(0).tolist()
        pairs = list(zip(labels, sims))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs

@dataclass
class ImageTags:
    colors: List[str]
    garments: List[str]
    patterns: List[str]
    styles: List[str]
    brand_hints: List[str]


def extract_tags(img: Image.Image, top_k: int = 3) -> ImageTags:
    colors = [c for c, _ in rank_labels(img, COLORS, prefix="a piece of clothing that is")[:top_k]]
    garments = [g for g, _ in rank_labels(img, GARMENTS)[:top_k]]
    patterns = [p for p, _ in rank_labels(img, PATTERNS)[:top_k]]
    styles = [s for s, _ in rank_labels(img, STYLES)[:2]]
    # Noisy, but can help: zero-shot brand hints
    brand_hints = [b for b, _ in rank_labels(img, BRANDS_COMMON, prefix="a clothing item by")[:2]]
    return ImageTags(colors, garments, patterns, styles, brand_hints)

# ----------------- Query builder -----------------
def build_queries(tags: ImageTags, show: str, movie: str, extra: str, max_q: int = 8) -> List[str]:
    ctx = " ".join([x for x in [show, movie, extra] if x]).strip()
    combos = []
    for c in (tags.colors or [""]):
        for g in (tags.garments or [""]):
            for ps in ((tags.patterns or []) + (tags.styles or []) or [""]):
                parts = [c, g, ps, ctx]
                q = " ".join([p for p in parts if p]).strip()
                if q:
                    combos.append(q)
    # brand-forward queries
    for b in (tags.brand_hints or []):
        if tags.garments:
            combos.insert(0, f"{b} {tags.garments[0]} {ctx}".strip())
    seen, out = set(), []
    for q in combos:
        if q not in seen:
            seen.add(q); out.append(q)
        if len(out) >= max_q:
            break
    if not out:
        base = " ".join([tags.colors[0] if tags.colors else "", tags.garments[0] if tags.garments else ""]).strip()
        if base:
            out = [base]
    return out

# ----------------- Site searchers -----------------

def parse_items_from_soup(soup: BeautifulSoup, selectors: List[Tuple[str, str]]) -> List[Hit]:
    hits: List[Hit] = []
    for sel, site in selectors:
        for node in soup.select(sel):
            a = node if node.name == "a" else node.select_one("a")
            if not a:
                continue
            href = a.get("href", "")
            title = (a.get_text(" ") or "").strip()
            if not href or not title:
                continue
            if href.startswith("/") and site.startswith("http"):
                href = site.rstrip("/") + href
            hits.append(Hit(source=site.split('//')[-1], title=title, url=href, snippet=""))
    return hits


def search_wornontv(q: str, limit: int = 15) -> List[Hit]:
    html = http_get("https://wornontv.net/", params={"s": q})
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    hits: List[Hit] = []
    for post in soup.select("article.post"):
        a = post.select_one("h2.entry-title a, .entry-title a, a")
        if not a:
            continue
        title = (a.get_text(" ") or "").strip()
        href = a.get("href", "")
        snippet_el = post.select_one(".entry-summary, .entry-excerpt, p")
        sn = (snippet_el.get_text(" ") if snippet_el else "").strip()
        hits.append(Hit("wornontv.net", title, href, sn))
        if len(hits) >= limit:
            break
    return hits


def search_shopyourtv(q: str, limit: int = 15) -> List[Hit]:
    html = http_get("https://shopyourtv.com/", params={"s": q})
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    hits: List[Hit] = []
    for post in soup.select("article, .post"):
        a = post.select_one("h2 a, h3 a, a")
        if not a:
            continue
        title = (a.get_text(" ") or "").strip()
        href = a.get("href", "")
        snippet_el = post.select_one(".entry-summary, .entry-content p, p")
        sn = (snippet_el.get_text(" ") if snippet_el else "").strip()
        hits.append(Hit("shopyourtv.com", title, href, sn))
        if len(hits) >= limit:
            break
    return hits


def search_spotern(q: str, limit: int = 15) -> List[Hit]:
    html = http_get("https://www.spotern.com/en/search", params={"q": q})
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    hits: List[Hit] = []
    for card in soup.select(".search-results .card, .result-list .card"):
        a = card.select_one("a[href]")
        if not a:
            continue
        href = a.get("href", "")
        title_el = card.select_one(".card-title, .title, h3, h2")
        title = (title_el.get_text(" ") if title_el else a.get("title") or a.get_text(" ")).strip()
        if href and not href.startswith("http"):
            href = "https://www.spotern.com" + href
        hits.append(Hit("spotern.com", title, href, ""))
        if len(hits) >= limit:
            break
    return hits


def search_thetake(q: str, limit: int = 15) -> List[Hit]:
    html = http_get("https://thetake.com/search", params={"q": q})
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    hits: List[Hit] = []
    for item in soup.select("a[href*='/product/'], a[href*='/scene/'], a[href*='/show/'], a[href*='/movie/']"):
        href = item.get("href", "")
        title = (item.get_text(" ") or "").strip()
        if href and not href.startswith("http"):
            href = "https://thetake.com" + href
        if title and href:
            hits.append(Hit("thetake.com", title, href, ""))
        if len(hits) >= limit:
            break
    return hits


def search_filmgarb(q: str, limit: int = 15) -> List[Hit]:
    html = http_get("https://filmgarb.com/search", params={"q": q})
    if not html:
        return []
    soup = BeautifulSoup(html, "html.parser")
    hits: List[Hit] = []
    for card in soup.select(".card, .search-result, article"):
        a = card.select_one("a[href]")
        if not a:
            continue
        href = a.get("href", "")
        title_el = card.select_one(".card-title, h2, h3")
        title = (title_el.get_text(" ") if title_el else a.get_text(" ")).strip()
        if href and not href.startswith("http"):
            href = "https://filmgarb.com" + href
        hits.append(Hit("filmgarb.com", title, href, ""))
        if len(hits) >= limit:
            break
    return hits

SITE_SEARCHERS = {
    "WornOnTV": search_wornontv,
    "ShopYourTV": search_shopyourtv,
    "Spotern": search_spotern,
    "TheTake": search_thetake,
    "Filmgarb": search_filmgarb,
}

# ----------------- Relevance filter -----------------
TOKEN_RE = re.compile(r"[^a-z0-9\s'&-]")

def tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = TOKEN_RE.sub(" ", s)
    return [t.strip("-'&") for t in s.split() if len(t.strip("-'&")) >= 3]


def score_hit(h: Hit, tokens: List[str]) -> int:
    hay = f"{h.title} {h.snippet}".lower()
    return sum(1 for t in tokens if t in hay)

# ----------------- Streamlit UI -----------------
st.set_page_config(page_title="TV/Movie Outfit Matcher", layout="wide")
st.title("🔎 TV/Movie Outfit Matcher")
st.caption("Upload your garment photos → get likely TV/Movie appearances from trusted 'seen-on-screen' sources.")

with st.sidebar:
    st.header("Options")
    show = st.text_input("Show (optional)")
    movie = st.text_input("Movie (optional)")
    extra = st.text_input("Extra keywords (character, brand guesses, etc.)")
    max_queries = st.slider("Max queries per image", 1, 12, 8)
    per_site = st.slider("Max results per site", 5, 40, 15)
    min_match = st.slider("Min token matches in title/snippet", 0, 3, 1)
    require_ctx = st.checkbox("Require show/movie tokens to appear", value=False)
    run_btn_area = st.empty()

uploads = st.file_uploader("Upload up to 5 clothing photos", type=["png", "jpg", "jpeg", "webp", "bmp"], accept_multiple_files=True)
if uploads:
    uploads = uploads[:5]

if uploads and run_btn_area.button("Run matcher", type="primary"):
    cols = st.columns(len(uploads))
    images: List[Image.Image] = []
    for i, up in enumerate(uploads):
        img = Image.open(up).convert("RGB")
        images.append(img)
        with cols[i]:
            st.image(img, caption=f"Image {i+1}", use_column_width=True)

    all_results: List[Hit] = []
    progress = st.progress(0, text="Analyzing images…")
    total_steps = len(images) * 2 + len(SITE_SEARCHERS)
    step = 0

    # Token setup
    show_tokens = tokenize(show) + tokenize(movie)
    other_tokens = tokenize(extra)
    all_tokens = list(dict.fromkeys(show_tokens + other_tokens))

    for idx, img in enumerate(images):
        # Tag extraction
        progress.progress(min((step:=step+1)/total_steps, 1.0), text=f"Analyzing image {idx+1} tags…")
        with st.spinner("CLIP tagging…"):
            tags = extract_tags(img, top_k=3)
        st.subheader(f"Image {idx+1} — predicted tags")
        st.write({
            "colors": tags.colors,
            "garments": tags.garments,
            "patterns": tags.patterns,
            "styles": tags.styles,
            "brand_hints": tags.brand_hints,
        })

        # Build queries
        qs = build_queries(tags, show, movie, extra, max_q=max_queries)
        st.caption("Generated queries:")
        st.code("\n".join(qs), language="text")

        progress.progress(min((step:=step+1)/total_steps, 1.0), text="Searching sources…")
        # Search each site with a short polite delay
        for sname, sfunc in SITE_SEARCHERS.items():
            try:
                hits = sfunc(qs[0])  # use the strongest query first per site
                # simple throttle
                time.sleep(0.6)
            except Exception:
                hits = []
            # Attach thumbnails lazily
            for h in hits:
                h.thumb = fetch_thumb(h.url) or ""
            all_results.extend(hits[:per_site])
            progress.progress(min((step:=step+1)/total_steps, 1.0), text=f"{sname} done")

    # De-dup by URL
    dedup: List[Hit] = []
    seen = set()
    for h in all_results:
        if h.url and h.url not in seen:
            seen.add(h.url); dedup.append(h)

    # Filter by relevance
    filtered: List[Hit] = []
    for h in dedup:
        score_all = score_hit(h, all_tokens)
        score_show = score_hit(h, show_tokens)
        if require_ctx and show_tokens and score_show == 0:
            continue
        if all_tokens and score_all < min_match:
            continue
        filtered.append(h)

    st.divider()
    st.subheader("Results")
    st.caption("Best‑effort matches — verify on the linked pages.")

    if not filtered:
        st.info("No strong matches. Try adding character/episode/brand in ‘Extra keywords’, or upload a clearer/closer image.")
    else:
        # Nice grid
        grid_cols = st.columns(3)
        for i, h in enumerate(filtered):
            with grid_cols[i % 3]:
                if h.thumb:
                    st.image(h.thumb, use_column_width=True)
                st.markdown(f"**[{h.title}]({h.url})**\n\n<small>{h.source}</small>", unsafe_allow_html=True)
                if h.snippet:
                    st.caption(h.snippet)

        # CSV download
        import csv
        from io import StringIO
        buf = StringIO()
        writer = csv.DictWriter(buf, fieldnames=["source", "title", "url", "snippet", "thumb"])
        writer.writeheader()
        for h in filtered:
            writer.writerow(asdict(h))
        st.download_button("Download CSV", buf.getvalue().encode("utf-8"), file_name="tv_movie_matches.csv", mime="text/csv")

st.divider()
st.caption("This tool queries public pages in a low‑rate, best‑effort way. Always respect each website’s Terms and use for research/personal workflows.")
