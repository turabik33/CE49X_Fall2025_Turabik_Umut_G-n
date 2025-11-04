import os, io, textwrap, time, random, string, base64
from datetime import datetime
import requests
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
from pytrends.request import TrendReq

# ================== KONFİG ==================
USE_MOCK_IMAGES = True

SEED_TERMS = ["minimal dövme"]
N_KEYWORDS_PER_SEED = 3
TIMEFRAME = "now 7-d"

LANG = "tr-TR"
GEO = "TR"


SAFE_FONT = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"  # Mac örneği
OUT_DIR = "out_tattoo"
RANDOM_BG_STYLES = [
    "soft natural daylight studio, beige fabric backdrop",
    "minimal editorial table setup, paper texture, neutral tones",
    "clean studio with subtle gradient background",
]

SD_API = "http://127.0.0.1:7860/sdapi/v1/txt2img"
SD_MODEL_HINT = "photorealistic studio macro skin photography"
SD_STEPS = 28
SD_CFG = 6.5
SD_WIDTH = 1024
SD_HEIGHT = 1280

HEADLINE_TEMPLATE = "{kw}"
CTA_TEXT = "Fikir keşfet"
TEXT_COLOR = (18, 18, 18)
PADDING = 48
MAX_HEADLINE_CHARS = 28

EXPORT_SIZES = {
    "square": (1080, 1080),
    "feed_vertical": (1080, 1350),
    "story": (1080, 1920),
}

BLOCKLIST = {"porn", "xxx", "18+", "erkeklere özel", "kadınlara özel", "+18"}
REPLACE = {"dovme": "dövme"}
# ============================================

def slugify(s: str) -> str:
    valid = "-_.%s%s" % (string.ascii_letters, string.digits)
    s = s.strip().replace(" ", "-")
    return "".join(ch for ch in s if ch in valid)[:80]

def get_top_keywords(seed_terms, n_per_seed=10, lang="tr-TR", geo="TR", timeframe="today 12-m"):
    # Google Trends API'den çok fazla 429 hatası alındığı için mock veri kullan
    print("⚠️  Google Trends API rate limit nedeniyle mock veri kullanılıyor...")
    
    # Dövme ile ilgili popüler anahtar kelimeler
    mock_keywords = [
        "minimal dövme", "ince çizgi dövme", "bilek dövmesi", "kol dövmesi", 
        "omuz dövmesi", "ayak bileği dövmesi", "çiçek dövme", "ejderha dövmesi",
        "geometrik dövme", "yazı dövmesi", "Roma rakamı dövmesi", "mandala dövme",
        "kelebek dövme", "gül dövme", "kartal dövme", "kurt dövme",
        "aile dövmesi", "çocuk dövmesi", "sevgi dövmesi", "umut dövmesi"
    ]
    
    # Seed terimlerine göre filtrele
    filtered = []
    for seed in seed_terms:
        seed_lower = seed.lower()
        for kw in mock_keywords:
            if any(word in kw.lower() for word in seed_lower.split()):
                filtered.append(kw)
    
    # Eğer hiç eşleşme yoksa genel dövme terimlerini döndür
    if not filtered:
        filtered = mock_keywords[:n_per_seed * len(seed_terms)]
    
    return filtered[:n_per_seed * len(seed_terms)]

def clean_keywords(kws):
    cleaned = []
    for kw in kws:
        kw_norm = kw.lower().strip()
        for a, b in REPLACE.items():
            kw_norm = kw_norm.replace(a, b)
        if any(bad in kw_norm for bad in BLOCKLIST):
            continue
        if 2 <= len(kw_norm) <= 40:
            cleaned.append(kw_norm)
    return list(dict.fromkeys(cleaned))

def make_prompt(keyword: str):
    bg = random.choice(RANDOM_BG_STYLES)
    skin_zones = [
        "close-up of wrist tattoo on forearm",
        "close-up of inner forearm tattoo",
        "close-up of ankle tattoo",
        "close-up of shoulder tattoo",
        "close-up of upper arm tattoo",
    ]
    modes = ["on_skin", "flash_sheet"]
    mode = random.choices(modes, weights=[0.7, 0.3], k=1)[0]

    if mode == "on_skin":
        zone = random.choice(skin_zones)
        prompt = (
            f"{SD_MODEL_HINT}, {zone}, fresh black ink, fine line style, "
            f"tattoo design concept: ({keyword}), {bg}, ultra-detailed skin texture, "
            f"85mm, f/2.8, soft even light, realistic pores, no glare, "
            f"no text overlay, no watermark"
        )
    else:
        prompt = (
            f"{SD_MODEL_HINT}, tattoo flash sheet on paper, flat lay top-down, "
            f"inked outlines of ({keyword}), neat composition, {bg}, "
            f"fine line, high resolution, no text overlay, no watermark"
        )

    negative = (
        "nudity, cleavage, breast, torso exposure, belly, thighs, "
        "lowres, blurry, deformed, extra fingers, watermark, logo, text, "
        "harsh shadows, overexposed, underexposed, duplicates"
    )
    return prompt, negative

def sd_txt2img(prompt: str, negative: str):
    # MOCK modu: gerçek model yoksa basit bir degrade zemin üret
    if globals().get("USE_MOCK_IMAGES", False):
        W, H = SD_WIDTH, SD_HEIGHT
        base = Image.new("RGB", (W, H))
        px = base.load()
        for y in range(H):
            v = int(230 - (y / H) * 40)  # 230 -> 190 arası gri degrade
            for x in range(W):
                px[x, y] = (v, v, v)
        return base

    # GERÇEK üretim: Stable Diffusion WebUI (Automatic1111) API
    payload = {
        "prompt": prompt,
        "negative_prompt": negative,
        "steps": SD_STEPS,
        "cfg_scale": SD_CFG,
        "width": SD_WIDTH,
        "height": SD_HEIGHT,
        "sampler_name": "DPM++ 2M Karras",
        "batch_size": 1,
        "n_iter": 1,
        "seed": -1,
    }
    r = requests.post(SD_API, json=payload, timeout=180)
    r.raise_for_status()
    data = r.json()
    img_b64 = data["images"][0]
    return Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")


def draw_text_center(img: Image.Image, headline: str, cta: str, font_path: str):
    W, H = img.size
    draw = ImageDraw.Draw(img)
    headline_size = max(48, int(W * 0.07))
    cta_size = max(28, int(W * 0.035))
    font_head = ImageFont.truetype(font_path, headline_size)
    font_cta = ImageFont.truetype(font_path, cta_size)

    wrapped = textwrap.fill(headline, width=16)
    w_h, h_h = draw.multiline_textbbox((0, 0), wrapped, font=font_head)[2:]
    w_c, h_c = draw.textbbox((0, 0), cta, font=font_cta)[2:]

    total_h = h_h + h_c + PADDING
    y0 = H - total_h - PADDING

    x_h = (W - w_h) // 2
    draw.multiline_text((x_h, y0), wrapped, font=font_head, fill=TEXT_COLOR, align="center")

    x_c = (W - w_c) // 2
    draw.text((x_c, y0 + h_h + int(PADDING * 0.3)), cta, font=font_cta, fill=TEXT_COLOR)
    return img

def export_sizes(img: Image.Image, base_path: str):
    results = []
    for name, size in EXPORT_SIZES.items():
        out = img.copy().resize(size, Image.LANCZOS)
        fp = f"{base_path}_{name}.jpg"
        out.save(fp, "JPEG", quality=92, optimize=True, progressive=True)
        results.append(fp)
    return results

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    kws = get_top_keywords(SEED_TERMS, n_per_seed=N_KEYWORDS_PER_SEED, lang=LANG, geo=GEO, timeframe=TIMEFRAME)
    kws = clean_keywords(kws)[:40]
    print(f"{len(kws)} anahtar kelime:", kws)

    rows = []
    for kw in kws:
        headline = HEADLINE_TEMPLATE.format(kw=kw[:MAX_HEADLINE_CHARS])
        prompt, negative = make_prompt(kw)
        try:
            img = sd_txt2img(prompt, negative)
            img = draw_text_center(img, headline=headline, cta=CTA_TEXT, font_path=SAFE_FONT)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            base = os.path.join(OUT_DIR, f"{slugify(kw)}_{ts}")
            outs = export_sizes(img, base)
            rows.append({"keyword": kw, "headline": headline, "prompt": prompt, "negative": negative, "outputs": "|".join(outs)})
            print("OK:", kw, "->", outs)
            time.sleep(2.0)  # Gecikmeyi 0.4'ten 2 saniyeye çıkar
        except Exception as e:
            print("FAIL:", kw, e)

    pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR, "run_report.csv"), index=False)
    print("Bitti:", OUT_DIR)

if __name__ == "__main__":
    main()
