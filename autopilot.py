"""yt-autopilot v2: trending topic -> story script -> Gemini images (consistent characters)
-> multi-voice TTS -> music + sfx -> Ken Burns video -> thumbnail -> scheduled upload -> dashboard data."""
import os, io, re, sys, ssl, glob, math, json, threading, itertools, time, base64, random, wave, asyncio, subprocess, traceback, datetime as dt
import requests, numpy as np
import edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass
LANG = os.getenv("CONTENT_LANG") or "hi"
REVIEW = (os.getenv("REVIEW_MODE") or "false").lower() == "true"
QUICK = (os.getenv("QUICK_TEST") or "").lower() == "true" or os.path.exists("quick_test.txt")  # fast smoke test
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SR = 24000
MUSIC_VOL = float(os.getenv("MUSIC_VOLUME") or 1.0)
SFX_VOL = float(os.getenv("SFX_VOLUME") or 1.0)
GEM = "https://generativelanguage.googleapis.com/v1beta/models/"


# ---------------------------------------------------------------- helpers
def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])


HARD = 120  # seconds: absolute limit for any single web request (requests' own timeout only counts silence)


def hard_post(url, hard=None, **kw):
    box = {}

    def go():
        try:
            box["r"] = requests.post(url, **kw)
        except Exception as e:
            box["e"] = e
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(hard or HARD)
    if t.is_alive():
        raise requests.exceptions.Timeout(f"hard timeout after {hard or HARD}s")
    if "e" in box:
        raise box["e"]
    return box["r"]


def dur(path):
    o = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                       capture_output=True, text=True, check=True)
    return float(o.stdout.strip())


def youtube():
    creds = Credentials(None, refresh_token=os.environ["YT_REFRESH_TOKEN"],
                        token_uri="https://oauth2.googleapis.com/token",
                        client_id=os.environ["YT_CLIENT_ID"], client_secret=os.environ["YT_CLIENT_SECRET"],
                        scopes=["https://www.googleapis.com/auth/youtube"])
    return build("youtube", "v3", credentials=creds)


_AV = None


def available_models():
    """Models this API key can call with generateContent (names change often, so ask Google)."""
    global _AV
    if _AV is None:
        _AV = []
        try:
            tok = None
            while True:
                params = {"key": os.environ["GEMINI_API_KEY"], "pageSize": 200}
                if tok:
                    params["pageToken"] = tok
                r = requests.get(GEM.rstrip("/"), params=params, timeout=60).json()
                for m in r.get("models", []):
                    if "generateContent" in m.get("supportedGenerationMethods", []):
                        _AV.append(m["name"].split("/")[-1])
                tok = r.get("nextPageToken")
                if not tok:
                    break
        except Exception as e:
            print("model list failed:", e)
        print("Available models:", [m for m in _AV if "flash" in m or "image" in m or "pro" in m][:40])
    return _AV


TEXT_PREF = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-3.5-flash", "gemini-3.8-flash", "gemini-3.5-flash-lite",
             "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-flash-latest",
             "gemini-2.5-flash", "gemini-2.5-flash-lite"]
IMG_PREF = ["gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-2.5-flash-image", "gemini-3-pro-image"]


def text_models():
    av = available_models()
    c = [os.getenv("GEMINI_MODEL")] if os.getenv("GEMINI_MODEL") else []
    c += [m for m in TEXT_PREF if m in av or not av]
    c += [m for m in av if re.fullmatch(r"gemini-[\d.]+-flash(-lite)?", m) and m not in c]
    return c


BAD_TEXT, GOOD_TEXT = set(), [None]


def gemini_json(prompt, temperature=0.8):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": temperature}}
    models = [m for m in text_models() if m not in BAD_TEXT]
    if GOOD_TEXT[0] in models:  # stick to the model that already worked
        models = [GOOD_TEXT[0]] + [m for m in models if m != GOOD_TEXT[0]]
    last = None
    for model in models:
        for i in range(2):
            try:
                r = hard_post(GEM + model + ":generateContent", params={"key": os.environ["GEMINI_API_KEY"]},
                                  json=body, timeout=100)
                if r.status_code in (400, 403, 404, 429):
                    last = f"{model}: {r.status_code} {r.text[:150]}"
                    print("text model skipped ->", last)
                    if r.status_code != 429:
                        BAD_TEXT.add(model)
                    break
                r.raise_for_status()
                parts = r.json()["candidates"][0]["content"]["parts"]
                txt = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                out = json.loads(txt)
                GOOD_TEXT[0] = model
                print("text model used:", model)
                return out
            except requests.exceptions.Timeout:
                last = f"{model}: timeout"
                print("text model timed out, dropping it:", model)
                BAD_TEXT.add(model)
                break
            except Exception as e:
                last = f"{model}: {e}"
                print("text model error:", last)
                time.sleep(4)
    raise RuntimeError(f"Gemini text failed on every model. Last: {last}")


IMG_MODELS = None


def gemini_image(prompt, refs, aspect):
    """Returns (mime, base64) or None. API images carry no visible watermark."""
    global IMG_MODELS
    if IMG_MODELS is None:
        av = available_models()
        IMG_MODELS = [os.getenv("GEMINI_IMAGE_MODEL")] if os.getenv("GEMINI_IMAGE_MODEL") else []
        IMG_MODELS += [m for m in IMG_PREF if m in av or not av]
        IMG_MODELS += [m for m in av if "image" in m and m not in IMG_MODELS]
        print("image models to try:", IMG_MODELS)
    parts = [{"text": prompt}] + [{"inlineData": {"mimeType": m, "data": d}} for m, d in refs]
    for model in list(IMG_MODELS):
        for attempt in range(3):
            cfg = {"responseModalities": ["IMAGE"]}
            if attempt != 1:
                cfg["imageConfig"] = {"aspectRatio": aspect}
            try:
                r = hard_post(GEM + model + ":generateContent", params={"key": os.environ["GEMINI_API_KEY"]},
                                  json={"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}, timeout=100)
            except Exception as e:
                print("img error, dropping model:", model, e)
                if model in IMG_MODELS:
                    IMG_MODELS.remove(model)
                break
            if r.status_code == 200:
                for p in r.json().get("candidates", [{}])[0].get("content", {}).get("parts", []):
                    d = p.get("inlineData") or p.get("inline_data")
                    if d:
                        return d.get("mimeType") or d.get("mime_type") or "image/png", d["data"]
                break  # no image returned -> next model
            print("img", model, r.status_code, r.text[:200])
            if r.status_code == 429 and "limit: 0" not in r.text:
                time.sleep(25); continue
            if r.status_code == 400 and attempt == 0:
                continue  # retry once without aspect config
            if r.status_code in (400, 403, 404, 429):
                if model in IMG_MODELS:
                    IMG_MODELS.remove(model)
                break
    return None


ALLOW_FALLBACK = (os.getenv("ALLOW_FALLBACK_IMAGES") or "").lower() == "true" or os.path.exists("allow_fallback.txt")


def pollinations_image(prompt, w, h):
    try:
        r = requests.get("https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt[:400]),
                         params={"width": w, "height": h, "nologo": "true", "model": "flux",
                                 "seed": random.randint(1, 99999)}, timeout=120)
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception as e:
        print("pollinations error:", e)
    return None


CF_ACCS = None


def _cf_accounts():
    """Up to 3 Cloudflare accounts (CF_ACCOUNT_ID/CF_API_TOKEN, ..._2, ..._3). When one runs out we move to the next."""
    global CF_ACCS
    if CF_ACCS is None:
        CF_ACCS = []
        for suf in ("", "_2", "_3"):
            acc = (os.getenv("CF_ACCOUNT_ID" + suf) or "").strip()
            tok = (os.getenv("CF_API_TOKEN" + suf) or "").strip()
            if acc and tok:
                CF_ACCS.append(dict(id=acc, tok=tok, dead=False, why="", fails=0, n=len(CF_ACCS) + 1))
        print(f"Cloudflare accounts configured: {len(CF_ACCS)}")
    return CF_ACCS


def cf_configured():
    return bool(_cf_accounts())


def cf_available():
    return any(not a["dead"] for a in _cf_accounts())


def cf_quota_exhausted():
    accs = _cf_accounts()
    return bool(accs) and all(a["dead"] and a["why"] == "quota" for a in accs)


def _cf_try(acc, prompt, steps):
    url = f"https://api.cloudflare.com/client/v4/accounts/{acc['id']}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    for attempt in range(3):
        body = {"prompt": prompt[:1900], "steps": steps}  # FLUX schnell makes square images; framing is done later
        try:
            r = hard_post(url, headers={"Authorization": f"Bearer {acc['tok']}"}, json=body, timeout=90)
        except Exception as e:
            acc["fails"] += 1
            print(f"cloudflare account {acc['n']} error:", e)
            if acc["fails"] >= 4:
                acc["dead"], acc["why"] = True, "errors"
                return None
            time.sleep(5)
            continue
        if r.status_code == 200:
            b64 = (r.json().get("result") or {}).get("image")
            if b64:
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            return None
        print(f"cloudflare account {acc['n']}", r.status_code, r.text[:160])
        if r.status_code in (401, 403):
            acc["dead"], acc["why"] = True, "auth"
            print(f"CLOUDFLARE account {acc['n']}: ID ya token galat hai (401/403).")
            return None
        if r.status_code == 429:
            low = r.text.lower()
            if "free allocation" in low or "neurons" in low:
                acc["dead"], acc["why"] = True, "quota"
                print(f"CLOUDFLARE account {acc['n']}: aaj ki free limit khatam -> agle account pe shift.")
                return None
            time.sleep(20)
    return None


def cloudflare_image(prompt, w=1024, h=1024, steps=4):
    for acc in _cf_accounts():
        if acc["dead"]:
            continue
        img = _cf_try(acc, prompt, steps)
        if img is not None:
            return img
        if not acc["dead"]:
            return None  # temporary problem, caller decides
    return None


# ---------------------------------------------------------------- analytics
def _secs(iso):
    m = re.match(r"P(?:(\d+)D)?T?(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", iso or "")
    if not m:
        return 0
    d, h, mi, s = [int(x or 0) for x in m.groups()]
    return d * 86400 + h * 3600 + mi * 60 + s


def own_videos(yt, n=50):
    ch = yt.channels().list(part="contentDetails", mine=True).execute()
    up = ch["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
    items = yt.playlistItems().list(part="contentDetails", playlistId=up, maxResults=n).execute()["items"]
    ids = [i["contentDetails"]["videoId"] for i in items]
    if not ids:
        return []
    res = yt.videos().list(part="snippet,statistics,contentDetails,status", id=",".join(ids)).execute()["items"]
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for v in res:
        pub = dt.datetime.fromisoformat(v["snippet"]["publishedAt"].replace("Z", "+00:00"))
        if pub > now or v["status"]["privacyStatus"] != "public":
            continue
        st = v["statistics"]
        out.append(dict(id=v["id"], title=v["snippet"]["title"], views=int(st.get("viewCount", 0)),
                        likes=int(st.get("likeCount", 0)), comments=int(st.get("commentCount", 0)),
                        age_days=(now - pub).days, hour=pub.astimezone(IST).hour,
                        short=0 < _secs(v["contentDetails"].get("duration")) <= 180))
    return out


def score(v):
    return v["views"] / max(1, min(v["age_days"], 30))


def mean(x):
    return sum(x) / len(x) if x else 0


def analyze(yt):
    vids = own_videos(yt)
    mature = [v for v in vids if v["age_days"] >= 3]
    hours = {}
    for v in mature:
        hours.setdefault(v["hour"], []).append(score(v))
    best = max(hours, key=lambda h: mean(hours[h])) if len(mature) >= 8 and hours else 19
    top = sorted(vids, key=score, reverse=True)[:8]
    return dict(videos=vids, mature=mature, best_hour=best,
                top=[(v["title"], v["views"], "short" if v["short"] else "long") for v in top])


def choose_format(stats):
    sh = [score(v) for v in stats["mature"] if v["short"]]
    lg = [score(v) for v in stats["mature"] if not v["short"]]
    if len(sh) < 3 or len(lg) < 3:
        return "short" if len(sh) <= len(lg) else "long"
    best, worse = ("short", "long") if mean(sh) >= mean(lg) else ("long", "short")
    return worse if random.random() < 0.25 else best


def next_slot(hour, extra_hours=0):
    now = dt.datetime.now(IST)
    t = now.replace(hour=hour, minute=0, second=0, microsecond=0)
    if t < now + dt.timedelta(minutes=45):
        t += dt.timedelta(days=1)
    t += dt.timedelta(hours=extra_hours)
    return t.astimezone(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def write_dashboard(stats, hist):
    sh = [score(v) for v in stats["mature"] if v["short"]]
    lg = [score(v) for v in stats["mature"] if not v["short"]]
    data = dict(updated=dt.datetime.now(IST).strftime("%d %b %Y %H:%M IST"), best_hour=stats["best_hour"],
                short=dict(n=len(sh), avg=round(mean(sh), 1)), long=dict(n=len(lg), avg=round(mean(lg), 1)),
                total_views=sum(v["views"] for v in stats["videos"]), videos=stats["videos"], history=hist[-30:])
    json.dump(data, open("data.json", "w"), ensure_ascii=False)


# ---------------------------------------------------------------- topic + script
def trending():
    yt_t, g_t = [], []
    try:
        g_t = __import__("pytrends.request", fromlist=["TrendReq"]).TrendReq(hl="hi-IN", tz=330) \
            .trending_searches(pn="india")[0].tolist()[:20]
    except Exception:
        pass
    return g_t


def pick_topic(yt, stats, fmt, hist):
    try:
        r = yt.videos().list(part="snippet", chart="mostPopular", regionCode="IN", maxResults=25).execute()
        yt_t = [i["snippet"]["title"] for i in r["items"]]
    except Exception:
        yt_t = []
    prompt = f"""You are the growth strategist of a YouTube channel (audience: India, language: {LANG}).
Pick ONE topic for a {'YouTube Short (~50 sec)' if fmt == 'short' else 'long video (5-7 min)'} that will be told as an illustrated story/explainer.
Channel's best videos so far (title, views, format): {stats['top'] or 'no data yet'}
Trending on YouTube India: {yt_t}
Trending on Google India: {trending()}
Already made (do NOT repeat): {[h['topic'] for h in hist[-40:]]}
Owner's niche hint: {os.getenv('NICHE_HINT') or 'none'}
Rules: prefer trending AND close to what already got views here; real verifiable facts only; avoid politics, religion disputes,
medical/financial advice, adult content, tragedies, real living people's private life, copyrighted characters.
Return JSON: {{"topic":"","angle":"","why":""}}"""
    return gemini_json(prompt)


def proofread(s):
    """Second Gemini pass: fix Hindi spelling/matra mistakes so subtitles and voice are correct."""
    if LANG != "hi":
        return s
    refs = [("title",), ("thumb_text",)] + [("line", i, j) for i, sc in enumerate(s["scenes"]) for j in range(len(sc["lines"]))]

    def get(r):
        return s[r[0]] if r[0] in ("title", "thumb_text") else s["scenes"][r[1]]["lines"][r[2]]["text"]
    texts = [get(r) for r in refs]
    try:
        out = gemini_json("Fix ONLY spelling, matra and grammar mistakes in these Hindi (Devanagari) strings. Keep meaning, "
                          "length, tone and any English words unchanged. Do not translate. Return JSON "
                          '{"strings": [...]} with exactly the same count and order.\n' + json.dumps(texts, ensure_ascii=False),
                          temperature=0.1)["strings"]
        if len(out) == len(texts):
            for r, t in zip(refs, out):
                if not isinstance(t, str) or not t.strip():
                    continue
                if r[0] in ("title", "thumb_text"):
                    s[r[0]] = t
                else:
                    s["scenes"][r[1]]["lines"][r[2]]["text"] = t
            print("proofread: ok")
    except Exception as e:
        print("proofread skipped:", e)
    return s


def write_script(t, fmt):
    short = fmt == "short"
    n = 6 if QUICK else (12 if short else 24)
    prompt = f"""Write a high-retention illustrated {'Short' if short else 'video'} for YouTube. Language of speech/title/description: {LANG}
(Hindi = Devanagari, simple spoken words). Topic: {t['topic']}. Angle: {t['angle']}.
{'Total spoken words 50-60 only (quick test), hook in first 2 seconds.' if QUICK else 'Total spoken words 110-130, hook in first 2 seconds.' if short else 'Total spoken words 800-950, hook in first 10 seconds, clear payoff, short CTA at end.'}
The VERY FIRST spoken line must be a hook under 12 words (shocking fact, bold claim or question), never a greeting like 'namaste dosto'. Accurate facts only, no clickbait lies, short punchy lines, a surprising fact or question every few seconds, end each scene on a mini-cliffhanger so viewers keep watching. Use 1-3 recurring characters ONLY if they make it better
(e.g. a curious kid + a wise teacher); narrator speaks the rest.
Make exactly {n} scenes. Each scene: "image_prompt" (English, vivid, describes action/setting/camera, NO text in image),
"characters" (ids appearing), "sfx" (one of none, whoosh, impact, ding), "lines" (1-3 items {{"speaker": "narrator" or character id, "text": "..."}}).
Characters: {{"id":"c1","name":"","look":"detailed fixed appearance: age, clothes, colors, hair","voice":"male|female|child|old"}}.
"style": ONE art style sentence reused for every image (e.g. "vibrant 3D animated film style, warm cinematic lighting").
"music_mood": one of calm, happy, sad, tense, epic.
SEO rules for YouTube search in India:
"title": max 65 chars. Start with the main search keyword people really type (in {LANG}; keep an English keyword if Indians search it in English), then a curiosity hook. Honest, no ALL CAPS, no emoji spam.
"description": FIRST 2 lines (max 150 chars total) = hook + main keyword (this is what search shows). Then a natural 100-180 word summary using related keywords. No links, no timestamps (added automatically). Last line: one friendly line asking to subscribe.
"hashtags": exactly 3 relevant hashtags. "tags": 12-15 strings (each max 30 chars) mixing broad and specific keywords, Hindi AND English/Hinglish spellings people search.
"thumb_text": max 3 words in {LANG}, a shocking number or curious question. "thumb_prompt": English, ONE dramatic close-up subject for the thumbnail illustration.
"category": one of education, science_tech, entertainment, howto.
{"" if short else '"chapters": 6-8 items {"scene": <1-based scene number where the chapter starts>, "title": "max 40 chars"}, the first one at scene 1.'}
Return JSON with keys: title, description, hashtags, tags, thumb_text, thumb_prompt, category, style, music_mood, characters, scenes{"" if short else ", chapters"}."""
    s = gemini_json(prompt)
    s = proofread(s)
    title = str(s.get("title", "")).strip()
    if len(title) > 70:
        title = title[:70].rsplit(" ", 1)[0]
    s["title"] = title
    tags, total = [], 0
    for tg in s.get("tags") or []:
        tg = str(tg).strip().lstrip("#")
        if tg and len(tg) <= 30 and tg.lower() not in [x.lower() for x in tags] and total + len(tg) < 450:
            tags.append(tg); total += len(tg) + 1
    s["tags"] = tags[:15]
    hs = [("#" + str(h).strip().lstrip("#").replace(" ", "")) for h in (s.get("hashtags") or [])[:3] if str(h).strip()]
    if short and "#Shorts" not in hs:
        hs.append("#Shorts")
    s["_hashtags"] = " ".join(hs)
    s["description"] = str(s.get("description", "")).strip()
    chars = []
    for i, c in enumerate(s.get("characters") or []):
        if isinstance(c, dict):
            c.setdefault("id", f"c{i + 1}")
            c["id"] = str(c["id"])
            c.setdefault("name", c["id"])
            c.setdefault("look", "")
            chars.append(c)
    s["characters"] = chars
    lookup = {}
    for c in chars:
        lookup[c["id"].lower()] = c["id"]
        lookup[str(c["name"]).lower()] = c["id"]

    def cid(x):  # Gemini sometimes returns names, ids or whole dicts
        if isinstance(x, dict):
            x = x.get("id") or x.get("name") or ""
        return lookup.get(str(x).strip().lower(), "")
    for sc in s["scenes"]:
        sc["characters"] = [c for c in (cid(x) for x in (sc.get("characters") or [])) if c]
        sc["lines"] = [l for l in sc.get("lines", []) if isinstance(l, dict) and l.get("text")] or [{"speaker": "narrator", "text": "..."}]
        for l in sc["lines"]:
            l["speaker"] = cid(l.get("speaker")) or "narrator"
    return s


# ---------------------------------------------------------------- images
def make_images(s, fmt, work):
    short = fmt == "short"
    W, H = (720, 1280) if short else (1280, 720)
    gw, gh = (576, 1024) if short else (1024, 576)  # generation size (FLUX likes multiples of 64)
    aspect = "9:16" if short else "16:9"
    style = s["style"]
    chars = {c["id"]: c for c in s.get("characters", [])}
    refs = {}
    n = len(s["scenes"])
    imgs = [None] * n
    use_gemini_refs = False
    if not cf_configured():
        print("NOTE: CF_ACCOUNT_ID / CF_API_TOKEN not set -> only paid Gemini or low-quality fallback will work")
    for i, sc in enumerate(s["scenes"]):
        who = [chars[c] for c in sc.get("characters", []) if c in chars]
        look = " ".join(f"{c['name']} is {c['look']}." for c in who)
        prompt = (f"{style}. {sc['image_prompt']}. {look} Highly detailed, sharp focus, professional quality, "
                  f"rich colors, cinematic composition. No text, no captions, no watermark, no logo.")
        img, src = cloudflare_image(prompt, gw, gh), "cloudflare"
        if img is None:
            r = gemini_image(prompt, [], aspect)
            if r:
                img, src = Image.open(io.BytesIO(base64.b64decode(r[1]))).convert("RGB"), "gemini"
        if img is None and ALLOW_FALLBACK:
            img, src = pollinations_image(prompt, gw, gh), "pollinations"
        print(f"scene {i + 1}/{n}: {'OK via ' + src if img else 'FAILED'}")
        imgs[i] = img
    good = [i for i, im in enumerate(imgs) if im is not None]
    if not good:
        raise RuntimeError(("Saare Cloudflare accounts ki aaj ki free image limit khatam hai (reset roz 5:30 AM IST). " if cf_quota_exhausted() else
                            "Koi image nahi ban payi. Cloudflare keys (CF_ACCOUNT_ID, CF_API_TOKEN) check karo. ")
                           + "Kharab video upload nahi kiya.")
    if len(good) < 0.6 * n:
        raise RuntimeError(f"Sirf {len(good)}/{n} images bani. Kharab video upload nahi kiya.")
    for i in range(n):  # fill gaps with nearest good image instead of a black card
        if imgs[i] is None:
            j = min(good, key=lambda g: abs(g - i))
            imgs[i] = imgs[j]
    paths = []
    for i, im in enumerate(imgs):
        p = f"img{i}.jpg"
        im.save(os.path.join(work, p), quality=95)
        paths.append(p)
    return paths


def make_thumb_image(s, work):
    """One extra picture made only for the thumbnail: single big subject, dramatic, high contrast."""
    if not cf_available():
        return None
    prompt = (f"{s['style']}. YouTube thumbnail illustration: {s.get('thumb_prompt') or s.get('title', '')}. "
              "Extreme close-up of ONE clear expressive subject, dramatic cinematic lighting, vibrant saturated colors, "
              "high contrast, simple dark uncluttered background, subject slightly right of centre, ultra detailed, sharp focus. "
              "No text, no watermark, no logo.")
    img = cloudflare_image(prompt)
    if img is None:
        return None
    p = "thumb_src.jpg"
    img.save(os.path.join(work, p), quality=95)
    return p


# ---------------------------------------------------------------- voices
VOICES = {
    "hi": {"narrator": ("hi-IN-MadhurNeural", "+0Hz", "-3%"), "male": ("hi-IN-MadhurNeural", "-8Hz", "+0%"),
           "female": ("hi-IN-SwaraNeural", "+0Hz", "-3%"), "child": ("hi-IN-SwaraNeural", "+55Hz", "+10%"),
           "old": ("hi-IN-MadhurNeural", "-25Hz", "-12%")},
    "en": {"narrator": ("en-IN-PrabhatNeural", "+0Hz", "-3%"), "male": ("en-IN-PrabhatNeural", "-8Hz", "+0%"),
           "female": ("en-IN-NeerjaNeural", "+0Hz", "-3%"), "child": ("en-IN-NeerjaNeural", "+55Hz", "+10%"),
           "old": ("en-IN-PrabhatNeural", "-25Hz", "-12%")},
}
# Gemini TTS (natural, expressive). Free tier is slow (~3 requests/min), so we pace calls.
GVOICE = {
    "narrator": ("Charon", "warm, engaging storyteller, natural conversational pace, clear pronunciation"),
    "male": ("Puck", "friendly, expressive, natural"),
    "female": ("Kore", "warm, expressive, natural"),
    "child": ("Leda", "playful child-like voice, high energy, youthful"),
    "old": ("Gacrux", "calm, mature, slow and wise"),
}
TTS_PREF = ["gemini-3.8-flash-tts", "gemini-3.8-flash-lite-tts", "gemini-3.1-flash-tts-preview", "gemini-2.5-flash-preview-tts"]
_TTS = None
_TTS_LAST = [0.0]
TTS_DEAD = False


def gemini_tts(text, kind, out_wav, work, style_override=None):
    """Writes a 24k mono wav (with 0.22s tail) and returns True, or False if Gemini TTS is unavailable."""
    global _TTS, TTS_DEAD
    if TTS_DEAD or (os.getenv("USE_GEMINI_TTS") or "true").lower() == "false":
        return False
    if _TTS is None:
        av = available_models()
        _TTS = [m for m in TTS_PREF if m in av] + [m for m in av if "tts" in m and m not in TTS_PREF]
        if not av:
            _TTS = TTS_PREF[:2]
        print("tts models to try:", _TTS)
    voice, style = GVOICE.get(kind, GVOICE["narrator"])
    style = style_override or style
    for model in list(_TTS):
        new = "3.8" in model
        for attempt in range(3):
            wait = 25 - (time.time() - _TTS_LAST[0])
            if wait > 0:
                time.sleep(wait)
            _TTS_LAST[0] = time.time()
            if new:
                part = {"text": text}
                if attempt != 1:
                    part["speech_metadata"] = {"style": style}
                vc = {"voice": voice}
            else:
                part = {"text": f"Say in a {style} tone: {text}"}
                vc = {"prebuiltVoiceConfig": {"voiceName": voice}}
            body = {"contents": [{"role": "user", "parts": [part]}],
                    "generationConfig": {"responseModalities": ["AUDIO"], "speechConfig": {"voiceConfig": vc}}}
            try:
                r = hard_post(GEM + model + ":generateContent", params={"key": os.environ["GEMINI_API_KEY"]},
                                  json=body, timeout=100)
            except Exception as e:
                print("tts error, dropping model:", model, e)
                if model in _TTS:
                    _TTS.remove(model)
                break
            if r.status_code == 200:
                try:
                    d = r.json()["candidates"][0]["content"]["parts"][0]
                    d = d.get("inlineData") or d.get("inline_data")
                    raw = base64.b64decode(d["data"])
                except Exception:
                    break
                src = os.path.join(work, "tts_raw.wav")
                if raw[:4] == b"RIFF":
                    open(src, "wb").write(raw)
                else:  # headerless 16-bit PCM, 24 kHz mono
                    with wave.open(src, "wb") as w:
                        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(raw)
                run(["ffmpeg", "-y", "-i", "tts_raw.wav", "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16",
                     "-af", "apad=pad_dur=0.22", os.path.basename(out_wav)], work)
                return True
            print("tts", model, r.status_code, r.text[:160])
            if r.status_code == 429 and "limit: 0" not in r.text and "per day" not in r.text.lower():
                time.sleep(30); continue
            if r.status_code == 400 and attempt == 0:
                continue
            if r.status_code in (400, 403, 404, 429):
                _TTS.remove(model)
                break
    if not _TTS:
        TTS_DEAD = True
    return False


async def _tts(text, v, path):
    kw = dict(rate=v[2], pitch=v[1])
    try:
        c = edge_tts.Communicate(text, v[0], boundary="WordBoundary", **kw)
    except TypeError:
        c = edge_tts.Communicate(text, v[0], **kw)
    words = []
    with open(path, "wb") as f:
        async for ch in c.stream():
            if ch["type"] == "audio":
                f.write(ch["data"])
            elif ch["type"] == "WordBoundary":
                words.append((ch["offset"] / 1e7, ch["duration"] / 1e7, ch["text"]))
    return words


def _ts(t):
    return f"{int(t // 3600):02d}:{int(t % 3600 // 60):02d}:{t % 60:06.3f}".replace(".", ",")


def approx_words(text, d):
    ws = text.split()
    if not ws:
        return []
    tot = sum(len(w) + 1 for w in ws)
    t, out = 0.05, []
    for w in ws:
        x = (d - 0.1) * (len(w) + 1) / tot
        out.append((t, x, w)); t += x
    return out


HOOK_STYLE = ("high-energy, punchy and dramatic opening line, slightly faster pace, attention-grabbing, "
              "like a top YouTube storyteller, natural spoken delivery")


def make_voice(s, work, chunk, use_gemini=True):
    kinds = {c["id"]: c.get("voice", "male") for c in s.get("characters", [])}
    vmap = VOICES.get(LANG, VOICES["hi"])
    cursor, wavs, caps, times = 0.0, [], [], []
    for si, sc in enumerate(s["scenes"]):
        start = cursor
        groups = []  # merge neighbouring lines of the same speaker -> fewer TTS requests
        for ln in sc["lines"]:
            if groups and groups[-1][0] == ln["speaker"]:
                groups[-1][1].append(ln["text"])
            else:
                groups.append([ln["speaker"], [ln["text"]]])
        for li, (sp, texts) in enumerate(groups):
            text = " ".join(texts)
            kind = kinds.get(sp, "narrator")  # unknown speaker name -> narrator
            hook = si == 0 and li == 0
            wav = f"l{si}_{li}.wav"
            words = None
            if use_gemini:
                if gemini_tts(text, kind, os.path.join(work, wav), work, HOOK_STYLE if hook else None):
                    pass
                else:
                    print("Gemini voice failed mid-video -> redoing whole voice with edge-tts (keeps one consistent voice)")
                    return make_voice(s, work, chunk, use_gemini=False)
            else:
                v = vmap.get(kind, vmap["narrator"])
                if hook:  # catchier opening: a bit faster and higher
                    v = (v[0], "+6Hz" if kind != "old" else v[1], "+8%")
                mp3 = f"l{si}_{li}.mp3"
                words = asyncio.run(_tts(text, v, os.path.join(work, mp3)))
                run(["ffmpeg", "-y", "-i", mp3, "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16",
                     "-af", "apad=pad_dur=0.22", wav], work)
            d = dur(os.path.join(work, wav))
            if not words:
                words = approx_words(text, d - 0.22)
            caps += [(cursor + w[0], cursor + w[0] + w[1], w[2]) for w in words]
            cursor += d
            wavs.append(os.path.join(work, wav))
        times.append((start, cursor))
        print(f"voice scene {si + 1}/{len(s['scenes'])} done ({'gemini' if use_gemini else 'edge'})")
    with wave.open(os.path.join(work, "voice.wav"), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(SR)
        for w in wavs:
            with wave.open(w) as r:
                out.writeframes(r.readframes(r.getnframes()))
    return times, caps


# ---------------------------------------------------------------- music + sfx (synthesised, copyright-free)
def read_wav(p):
    with wave.open(p) as w:
        return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768


def write_wav(p, x):
    with wave.open(p, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((np.clip(x, -1, 1) * 32767).astype(np.int16).tobytes())


def _band(x, lo, hi):
    F = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    return np.fft.irfft(F * ((f >= lo) & (f < hi)), n=len(x))


def sfx(kind):
    """Small, soft, synthesised effects (original, copyright-free)."""
    if kind == "whoosh":  # rising band-passed noise sweep
        n = int(0.8 * SR)
        x = np.random.randn(n)
        t = np.linspace(0, 1, n)
        out = np.zeros(n)
        for k, (lo, hi) in enumerate(((150, 600), (600, 2000), (2000, 6500))):
            out += _band(x, lo, hi) * np.exp(-((t - (0.25 + 0.25 * k)) ** 2) / (2 * 0.12 ** 2))
        out *= np.sin(np.pi * t) ** 1.5
        return out / max(1e-6, np.abs(out).max()) * 0.7
    t = np.arange(int(1.3 * SR)) / SR
    if kind == "impact":  # deep boom + short crack
        boom = np.sin(2 * np.pi * np.cumsum(38 + 70 * np.exp(-t * 10)) / SR) * np.exp(-t * 3.2)
        crack = np.random.randn(len(t)) * np.exp(-t * 45) * 0.35
        return (boom + crack) * 0.9
    if kind == "ding":  # bell-like inharmonic partials
        y = sum(a * np.sin(2 * np.pi * 988 * r * t) * np.exp(-d * t) for r, a, d in ((1, 1.0, 3.0), (2.76, 0.45, 5.0), (5.4, 0.25, 8.0)))
        return y * 0.45
    return None


CHORDS = {"calm": [220, 277.2, 329.6], "happy": [261.6, 329.6, 392], "sad": [220, 261.6, 329.6],
          "tense": [146.8, 174.6, 220], "epic": [130.8, 196, 261.6, 392]}


def pad(mood, total):
    """Fallback music: soft pad + gentle arpeggio + echo (used only if no music/ tracks are uploaded)."""
    n = int(total * SR)
    x = np.arange(n) / SR
    chord = CHORDS.get(mood, CHORDS["calm"])
    m = np.zeros(n)
    for i, f in enumerate(chord):
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.07 + 0.03 * i) * x + i)
        m += (np.sin(2 * np.pi * f * x) + 0.3 * np.sin(2 * np.pi * 2 * f * x)) * lfo
    m /= max(1e-6, np.abs(m).max())
    bpm = {"calm": 72, "happy": 108, "sad": 60, "tense": 92, "epic": 84}.get(mood, 80)
    step = 60 / bpm / 2
    seq = [f * 2 for f in chord] + [f * 4 for f in chord[::-1]]
    arp = np.zeros(n)
    tt = np.arange(int(1.6 * SR)) / SR
    t0, i = 0.0, 0
    while t0 < total - 1:
        f = seq[i % len(seq)]
        note = (np.sin(2 * np.pi * f * tt) + 0.4 * np.sin(2 * np.pi * 2 * f * tt) * np.exp(-tt * 5)) * np.exp(-tt * 3.2)
        st = int(t0 * SR)
        arp[st:st + len(note)] += note[:n - st]
        t0 += step; i += 1
    arp /= max(1e-6, np.abs(arp).max())
    mix = m * 0.6 + arp * 0.45
    for dly, g in ((int(0.375 * SR), 0.35), (int(0.75 * SR), 0.18)):
        mix[dly:] += g * mix[:-dly].copy()
    mix /= max(1e-6, np.abs(mix).max())
    fade = np.minimum(1, np.minimum(x / 2, (total - x) / 2))
    return mix * fade


SCALES = {"major": [0, 2, 4, 5, 7, 9, 11], "minor": [0, 2, 3, 5, 7, 8, 10]}
MOODS = {
    "calm": dict(bpm=76, root=57, scale="minor", prog=[0, 5, 3, 4], sev=True, swing=0.14, drums=1, mel=0.55, lp=4200, crackle=True),
    "happy": dict(bpm=104, root=60, scale="major", prog=[0, 4, 5, 3], sev=False, swing=0.06, drums=2, mel=0.75, lp=8000, crackle=False),
    "sad": dict(bpm=64, root=57, scale="minor", prog=[0, 3, 5, 4], sev=True, swing=0.1, drums=0, mel=0.35, lp=3800, crackle=True),
    "tense": dict(bpm=92, root=50, scale="minor", prog=[0, 0, 5, 4], sev=False, swing=0.0, drums=2, mel=0.2, lp=6000, crackle=False),
    "epic": dict(bpm=88, root=55, scale="minor", prog=[0, 5, 2, 6], sev=False, swing=0.0, drums=3, mel=0.45, lp=9000, crackle=False),
}


def _mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def _tt(d):
    return np.arange(int(d * SR)) / SR


def _kick():
    t = _tt(0.35)
    return np.sin(2 * np.pi * np.cumsum(45 + 95 * np.exp(-t * 32)) / SR) * np.exp(-t * 8)


def _snare():
    t = _tt(0.25)
    n = np.diff(np.random.randn(len(t)), prepend=0)
    return n * 0.5 * np.exp(-t * 22) + np.sin(2 * np.pi * 185 * t) * 0.35 * np.exp(-t * 30)


def _hat():
    t = _tt(0.06)
    return np.diff(np.random.randn(len(t)), prepend=0) * np.exp(-t * 70) * 0.35


def _epiano(f, d):
    t = _tt(d)
    mod = np.sin(2 * np.pi * f * t) * 1.4 * np.exp(-5 * t)
    y = np.sin(2 * np.pi * f * t + mod) * np.exp(-1.8 * t) + 0.2 * np.sin(2 * np.pi * 3 * f * t) * np.exp(-9 * t)
    return y * np.minimum(1, t * 120)


def _bass(f, d):
    t = _tt(d)
    return (np.sin(2 * np.pi * f * t) + 0.25 * np.sin(2 * np.pi * 2 * f * t)) * np.exp(-1.6 * t) * np.minimum(1, t * 150)


def _pluck(f, d):
    t = _tt(d)
    return (np.sin(2 * np.pi * f * t) + 0.35 * np.sin(2 * np.pi * 2 * f * t) * np.exp(-6 * t)) * np.exp(-4 * t) * np.minimum(1, t * 200)


def lofi_loop(mood, seed):
    """8 bars of original lo-fi/ambient music (chords, bass, drums, melody, reverb). Copyright-free, made from scratch."""
    rng = random.Random(seed)
    np.random.seed(seed)
    c = MOODS.get(mood, MOODS["calm"])
    beat = 60 / c["bpm"]
    bar = 4 * beat
    L = int(8 * bar * SR)
    tail = 3 * SR
    buf = np.zeros(L + tail)

    def add(snd, t, g=1.0):
        st = int(t * SR)
        if st < L:
            e = min(len(buf), st + len(snd))
            buf[st:e] += snd[:e - st] * g

    sc = SCALES[c["scale"]]

    def note(deg, octv=0):
        return c["root"] + sc[deg % 7] + 12 * (deg // 7) + 12 * octv

    for ci, deg in enumerate(c["prog"]):
        t0 = ci * 2 * bar
        top = deg + 6 if c["sev"] else deg + 7
        for j, m in enumerate([note(deg), note(deg + 2), note(deg + 4), note(top)]):
            add(_epiano(_mtof(m), 2 * bar + 0.5), t0 + j * 0.018, 0.20)
        for b in (0, 3, 4, 6):
            add(_bass(_mtof(note(deg) - 12), beat * 1.9), t0 + b * beat, 0.55)
    for b in range(8):
        t = b * bar
        if c["drums"] >= 1:
            add(_kick(), t, 0.8)
            add(_kick(), t + 2 * beat, 0.7)
            if c["drums"] >= 2:
                add(_kick(), t + 2.5 * beat, 0.45)
            if c["drums"] >= 3:
                add(_kick(), t + beat, 0.5); add(_kick(), t + 3 * beat, 0.5)
            add(_snare(), t + beat, 0.5)
            add(_snare(), t + 3 * beat, 0.55)
            for e in range(8):
                add(_hat(), t + e * beat / 2 + (c["swing"] * beat if e % 2 else 0), 0.5 if e % 2 == 0 else 0.3)
    pent, cur = [0, 1, 2, 4, 5], 2
    for b in range(8):
        for e in range(8):
            if rng.random() < c["mel"] * (1 if e % 2 == 0 else 0.5):
                cur = max(0, min(4, cur + rng.choice([-2, -1, 0, 1, 2])))
                add(_pluck(_mtof(note(pent[cur], 1)), 0.9), b * bar + e * beat / 2 + (c["swing"] * beat if e % 2 else 0), 0.16)
    loop = buf[:L].copy()
    loop[:tail] += buf[L:]
    n = int(1.1 * SR)
    ir = np.random.randn(n) * np.exp(-np.arange(n) / (0.28 * SR))
    ir /= np.sqrt(np.sum(ir ** 2))
    wet = np.fft.irfft(np.fft.rfft(loop) * np.fft.rfft(ir, n=L), n=L)
    x = loop * 0.8 + wet * 0.5
    f = np.fft.rfftfreq(L, 1 / SR)
    x = np.fft.irfft(np.fft.rfft(x) / (1 + (f / c["lp"]) ** 2), n=L)
    if c["crackle"]:
        x += (np.random.rand(L) < 0.0004) * np.random.randn(L) * 0.12 + np.random.randn(L) * 0.004
    return x / max(1e-6, np.abs(x).max()) * 0.9


def lofi(mood, total):
    seeds = random.sample(range(1, 10 ** 6), 2)
    A, B = lofi_loop(mood, seeds[0]), lofi_loop(mood, seeds[1])
    n = int(total * SR)
    out, pos, i = np.zeros(n), 0, 0
    while pos < n:
        seg = A if i % 2 == 0 else B
        take = min(len(seg), n - pos)
        out[pos:pos + take] = seg[:take]
        pos += take; i += 1
    x = np.arange(n) / SR
    return out * np.minimum(1, np.minimum(x / 1.5, (total - x) / 2.5))


def _load_track(path, total, work, label):
    out = os.path.join(work, "music.wav")
    try:
        run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", os.path.abspath(path), "-t", f"{total:.2f}", "-ar", str(SR),
             "-ac", "1", "-sample_fmt", "s16", "-af", f"afade=t=in:d=2,afade=t=out:st={max(0, total - 3):.2f}:d=3", out])
        x = read_wav(out)
        print("music track:", label)
        return x / max(1e-6, np.abs(x).max())
    except Exception as e:
        print("music file failed:", e)
        return None


def user_music(mood, total, work):
    """Optional: real tracks put in the music/ folder (name them like calm_1.mp3, happy_2.mp3 ...)."""
    files = [f for f in glob.glob("music/*") if f.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".aac"))]
    if not files:
        return None
    pick = [f for f in files if mood in os.path.basename(f).lower()] or files
    f = random.choice(pick)
    return _load_track(f, total, work, os.path.basename(f))


LYRIA_PROMPTS = {
    "calm": "relaxed warm lo-fi hip hop beat with soft electric piano, mellow bass and gentle drums",
    "happy": "upbeat cheerful bright instrumental with light percussion, plucky melody and a feel-good groove",
    "sad": "gentle emotional piano with soft strings, slow tempo",
    "tense": "dark suspenseful cinematic pulse with low strings and subtle percussion, slowly building tension",
    "epic": "epic cinematic orchestral instrumental with big drums, strings and rising energy",
}
LYRIA_DEAD = False


def lyria_music(mood, total, work):
    """Gemini Lyria 3 (real AI-composed music). If it is not available on this key, we silently fall back."""
    global LYRIA_DEAD
    if LYRIA_DEAD or (os.getenv("USE_LYRIA") or "true").lower() == "false":
        return None
    av = available_models()
    models = [m for m in ("lyria-3-pro-preview", "lyria-3-clip-preview") if m in av]
    if total <= 35 and "lyria-3-clip-preview" in models:
        models = ["lyria-3-clip-preview"] + [m for m in models if m != "lyria-3-clip-preview"]
    prompt = (f"Instrumental background music for a YouTube explainer video: {LYRIA_PROMPTS.get(mood, LYRIA_PROMPTS['calm'])}. "
              "NO vocals, no lyrics, no spoken words. Steady energy without sudden changes, easy to loop, mixed quietly "
              "so a narrator can talk over it. About 90 seconds long.")
    for model in models:
        try:
            r = hard_post(GEM + model + ":generateContent", params={"key": os.environ["GEMINI_API_KEY"]}, hard=240,
                          json={"contents": [{"parts": [{"text": prompt}]}],
                                "generationConfig": {"responseModalities": ["AUDIO", "TEXT"]}}, timeout=240)
        except Exception as e:
            print("lyria error:", model, e)
            continue
        if r.status_code != 200:
            print("lyria", model, r.status_code, r.text[:140])
            continue
        for p in r.json().get("candidates", [{}])[0].get("content", {}).get("parts", []):
            d = p.get("inlineData") or p.get("inline_data")
            if d and "audio" in (d.get("mimeType") or d.get("mime_type") or ""):
                path = os.path.join(work, "lyria_raw.mp3" if "wav" not in (d.get("mimeType") or "") else "lyria_raw.wav")
                open(path, "wb").write(base64.b64decode(d["data"]))
                x = _load_track(path, total, work, "Lyria " + model)
                if x is not None:
                    return x
    LYRIA_DEAD = True
    return None


def mix_audio(s, times, work):
    voice = read_wav(os.path.join(work, "voice.wav"))
    total = len(voice) / SR + 0.6
    voice = np.pad(voice, (0, int(total * SR) - len(voice)))
    voice /= max(1e-6, np.abs(voice).max()) / 0.8  # even loudness whatever the TTS returned
    k = int(0.25 * SR)
    cs = np.cumsum(np.insert(np.abs(voice), 0, 0))
    env = np.pad((cs[k:] - cs[:-k]) / k, (k // 2, len(voice) - (len(cs) - k) - k // 2))[:len(voice)]
    duck = 1 - 0.7 * np.clip(env * 12, 0, 1)
    mood = s.get("music_mood", "calm")
    track = user_music(mood, total, work)
    if track is None:
        track = lyria_music(mood, total, work)
    real = track is not None
    if track is None:
        try:
            track = lofi(mood, total)
            print("music: generated", mood, "lo-fi track")
        except Exception as e:
            print("lofi failed, using simple pad:", e)
            track = pad(mood, total)
    track = np.pad(track, (0, max(0, len(voice) - len(track))))[:len(voice)]
    bed = track * (0.15 if real else 0.12) * MUSIC_VOL * duck
    fx = np.zeros(len(voice))
    for i, (a, _) in enumerate(times):
        for kind, off in (("whoosh", -0.15 if i else None), (s["scenes"][i].get("sfx", "none"), 0.05)):
            if off is None:
                continue
            snd = sfx(kind)
            if snd is None:
                continue
            st = max(0, int((a + off) * SR))
            fx[st:st + len(snd)] += snd[:len(fx) - st]
    mix = voice + bed + fx * 0.22 * SFX_VOL
    mix /= max(1.0, np.abs(mix).max() / 0.95)
    write_wav(os.path.join(work, "mix.wav"), mix)


# ---------------------------------------------------------------- video
FPS = 30
try:
    _TR, _BC = Image.Transform.EXTENT, Image.Resampling.BICUBIC
except AttributeError:
    _TR, _BC = Image.EXTENT, Image.BICUBIC


def _ass_time(t):
    cs = int(round(max(0, t) * 100))
    return f"{cs // 360000}:{cs % 360000 // 6000:02d}:{cs % 6000 // 100:02d}.{cs % 100:02d}"


def make_ass(path, caps, W, H, short, hook):
    """Word-highlight captions (libass shapes Devanagari correctly) + a hook line at the top."""
    fs = 88 if short else 54
    head = (f"[Script Info]\nScriptType: v4.00+\nPlayResX: {W}\nPlayResY: {H}\nWrapStyle: 0\nScaledBorderAndShadow: yes\n\n"
            "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, "
            "Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
            f"Style: Cap,Noto Sans Devanagari,{fs},&H00FFFFFF,&H00FFFFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{6 if short else 4},2,2,60,60,{int(H * 0.22) if short else 50},1\n"
            f"Style: Hook,Noto Sans Devanagari,{int(fs * 1.2)},&H0000E5FF,&H0000E5FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,{7 if short else 5},3,8,60,60,{int(H * 0.09)},1\n\n"
            "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n")
    ev = []
    if hook:
        ev.append(f"Dialogue: 1,{_ass_time(0.2)},{_ass_time(2.8)},Hook,,0,0,0,,{{\\fad(250,350)}}{hook}")
    N = 3 if short else 5
    chunks, cur = [], []
    for w in caps:
        cur.append(w)
        if len(cur) >= N or w[2].rstrip()[-1:] in "\u0964.?!":
            chunks.append(cur); cur = []
    if cur:
        chunks.append(cur)
    for ci, ch in enumerate(chunks):
        nxt = chunks[ci + 1][0][0] if ci + 1 < len(chunks) else ch[-1][1] + 0.4
        for k, w in enumerate(ch):
            end = ch[k + 1][0] if k + 1 < len(ch) else min(nxt, max(w[1], w[0] + 0.15) + 0.25)
            parts = []
            for j, x in enumerate(ch):
                x = x[2].replace("{", "").replace("}", "")
                parts.append(("{\\c&H0000E5FF&}" + x + "{\\c&H00FFFFFF&}") if j == k else x)
            pop = "{\\fscx100\\fscy100\\t(0,90,\\fscx108\\fscy108)\\t(90,180,\\fscx100\\fscy100)}"
            ev.append(f"Dialogue: 0,{_ass_time(w[0])},{_ass_time(end)},Cap,,0,0,0,,{pop}{' '.join(parts)}")
    open(path, "w", encoding="utf-8").write(head + "\n".join(ev) + "\n")


def _prep(path, W, H, k=1.3):
    im = Image.open(path).convert("RGB")
    tw, th = int(W * k), int(H * k)
    if im.width / im.height > (W / H) * 1.25:
        # square/wide source on a vertical Short: keep the WHOLE picture, fill the rest with a blurred copy
        r = max(tw / im.width, th / im.height)
        bg = im.resize((int(im.width * r + 0.5), int(im.height * r + 0.5)), Image.BILINEAR)
        l, t = (bg.width - tw) // 2, (bg.height - th) // 2
        bg = bg.crop((l, t, l + tw, t + th)).filter(ImageFilter.GaussianBlur(28))
        bg = ImageEnhance.Brightness(bg).enhance(0.55)
        fh = int(im.height * tw / im.width)
        fg = im.resize((tw, fh), Image.LANCZOS).filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))
        bg.paste(fg, (0, max(0, int(th * 0.44 - fh / 2))))  # slightly above centre: captions sit on the lower blur
        return bg
    r = max(tw / im.width, th / im.height)
    im = im.resize((max(tw, int(im.width * r + 0.5)), max(th, int(im.height * r + 0.5))), Image.LANCZOS)
    l, t = (im.width - tw) // 2, (im.height - th) // 2
    return im.crop((l, t, l + tw, t + th)).filter(ImageFilter.UnsharpMask(radius=2, percent=60, threshold=3))


def _box(tw, th, mode, sign, t):
    e = t * t * (3 - 2 * t)  # ease in-out
    z = 1.0 + 0.16 * e if mode == 0 else (1.16 - 0.16 * e if mode == 1 else 1.14)
    vw, vh = tw / z, th / z
    cx, cy = tw / 2, th / 2
    if mode == 2:
        cx += sign * (e - 0.5) * (tw - vw) * 0.9
    if mode == 3:
        cy += sign * (e - 0.5) * (th - vh) * 0.9
    return (cx - vw / 2, cy - vh / 2, cx + vw / 2, cy + vh / 2)


def render(s, fmt, work, imgs, times, caps):
    short = fmt == "short"
    W, H = (1080, 1920) if short else (1280, 720)
    preps = [_prep(os.path.join(work, p), W, H) for p in imgs]
    A, B = [a for a, _ in times], [b for _, b in times]
    total = B[-1] + 0.6
    B[-1] = total
    h = 0.175  # half of the 0.35s crossfade
    MODES = [0, 2, 1, 3]

    def scene_frame(i, gt):
        t = min(1.0, max(0.0, (gt - A[i]) / max(0.1, B[i] - A[i])))
        tw, th = preps[i].size
        box = _box(tw, th, MODES[i % 4], 1 if i % 2 == 0 else -1, t)
        return preps[i].transform((W, H), _TR, box, resample=_BC)

    def frame_at(gt):
        i = 0
        while i < len(A) - 1 and gt >= B[i] - h:
            i += 1
        if i > 0 and gt < A[i] + h:
            al = min(1.0, max(0.0, (gt - (A[i] - h)) / (2 * h)))
            return Image.blend(scene_frame(i - 1, gt), scene_frame(i, gt), al)
        return scene_frame(i, gt)

    n = int(math.ceil(total * FPS))
    cmd = ["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p", "joined.mp4"]
    log = open(os.path.join(work, "ff_frames.log"), "w")
    p = subprocess.Popen(cmd, cwd=work, stdin=subprocess.PIPE, stderr=log)
    for k in range(n):
        p.stdin.write(frame_at(k / FPS).tobytes())
        if k % 300 == 0:
            print(f"render {k}/{n} frames", flush=True)
    p.stdin.close()
    if p.wait():
        raise RuntimeError("frame encode failed: " + open(os.path.join(work, "ff_frames.log")).read()[-800:])
    vf = "eq=contrast=1.05:saturation=1.12,vignette=angle=PI/7"
    if caps:
        make_ass(os.path.join(work, "caps.ass"), caps, W, H, short, s.get("thumb_text", ""))
        vf += ",ass=caps.ass"
    run(["ffmpeg", "-y", "-i", "joined.mp4", "-i", "mix.wav", "-vf", vf, "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-af", "loudnorm=I=-14:TP=-1.5:LRA=11", "-ar", "48000", "-ac", "2", "-c:a", "aac", "-b:a", "192k",
         "-pix_fmt", "yuv420p", "-movflags", "+faststart",
         "-shortest", "final.mp4"], work)
    return os.path.join(work, "final.mp4")


_FONTS = {}


def _font(size, latin=False):
    key = (size, latin)
    if key not in _FONTS:
        try:
            q = "Noto Sans:bold" if latin else "Noto Sans Devanagari:bold"
            f = subprocess.run(["fc-match", "-f", "%{file}", q], capture_output=True, text=True).stdout
            _FONTS[key] = ImageFont.truetype(f, size)
        except Exception:
            _FONTS[key] = ImageFont.load_default()
    return _FONTS[key]


def _runs(text):
    return re.findall(r"[\u0900-\u097F]+|[^\u0900-\u097F]+", text)  # Hindi runs vs English/digit runs (different fonts)


def _tlen(d, text, size):
    return sum(d.textlength(r, font=_font(size, latin=not re.match(r"[\u0900-\u097F]", r))) for r in _runs(text))


def _draw_runs(d, x, y, text, size, fill, sw, sf):
    for r in _runs(text):
        f = _font(size, latin=not re.match(r"[\u0900-\u097F]", r))
        d.text((x, y), r, font=f, fill=fill, stroke_width=sw, stroke_fill=sf)
        x += d.textlength(r, font=f)


def _thumb_layout(d, text, box_w, max_h):
    """Pick line breaks + the biggest font size that fits the left text area."""
    words = text.split() or [text]
    best = None
    for n in (1, 2, 3):
        if n > len(words):
            break
        for cuts in itertools.combinations(range(1, len(words)), n - 1):
            idx = (0,) + cuts + (len(words),)
            lines = [" ".join(words[idx[i]:idx[i + 1]]) for i in range(n)]
            for size in range(170, 58, -6):
                if int(size * 1.3) * n <= max_h and max(_tlen(d, ln, size) for ln in lines) <= box_w:
                    cand = (size, -n, lines)
                    if best is None or cand[:2] > best[:2]:
                        best = cand
                    break
    return best or (60, -3, [" ".join(words[:2]), " ".join(words[2:4]), " ".join(words[4:6])])


def make_thumb(img_path, text, out):
    """1280x720: sharp, colourful subject on the right, big high-contrast text on the left (typical high-CTR layout)."""
    W, H = 1280, 720
    src = Image.open(img_path).convert("RGB")
    r = max(W / src.width, H / src.height)
    bg = src.resize((int(src.width * r) + 1, int(src.height * r) + 1), Image.BILINEAR)
    l, t = (bg.width - W) // 2, (bg.height - H) // 2
    bg = ImageEnhance.Brightness(bg.crop((l, t, l + W, t + H)).filter(ImageFilter.GaussianBlur(22))).enhance(0.5)
    side = min(src.width, src.height)
    sq = src.crop(((src.width - side) // 2, (src.height - side) // 2, (src.width + side) // 2, (src.height + side) // 2))
    fg = sq.resize((H, H), Image.LANCZOS)
    fg = ImageEnhance.Color(ImageEnhance.Contrast(fg).enhance(1.18)).enhance(1.3).filter(ImageFilter.UnsharpMask(2, 80, 3))
    ramp = (np.clip(np.arange(H) / 220.0, 0, 1) * 255).astype(np.uint8)
    bg.paste(fg, (W - H, 0), Image.fromarray(np.tile(ramp, (H, 1)), "L"))
    shade = (np.clip(1 - np.arange(W) / 780.0, 0, 1) ** 1.2 * 0.8).astype(np.float32)
    arr = np.asarray(bg).astype(np.float32) * (1 - shade)[None, :, None]
    img = ImageEnhance.Contrast(Image.fromarray(arr.astype(np.uint8))).enhance(1.08)
    d = ImageDraw.Draw(img)
    size, _, lines = _thumb_layout(d, text, 640, int(H * 0.8))
    lh = int(size * 1.3)
    y = (H - lh * len(lines)) // 2
    for i, ln in enumerate(lines):
        col = "#FFFFFF" if i % 2 == 0 else "#FFD400"
        _draw_runs(d, 55 + 6, y + 6, ln, size, "#000000", 10, "#000000")  # shadow
        _draw_runs(d, 55, y, ln, size, col, 10, "#000000")
        y += lh
    for q in (92, 85, 78):
        img.save(out, "JPEG", quality=q)
        if os.path.getsize(out) < 1_900_000:  # YouTube limit is 2 MB
            break
    return out


# ---------------------------------------------------------------- upload
def upload(yt, video, thumb, s, publish_at):
    status = {"privacyStatus": "private", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True}
    if publish_at:
        status["publishAt"] = publish_at
    body = {"snippet": {"title": s["title"], "description": s["description"], "tags": s["tags"][:15],
                        "categoryId": {"education": "27", "science_tech": "28", "entertainment": "24", "howto": "26"}.get(str(s.get("category")), "27"), "defaultLanguage": LANG, "defaultAudioLanguage": LANG}, "status": status}
    for attempt in (0, 1):
        try:
            req = yt.videos().insert(part="snippet,status", body=body,
                                     media_body=MediaFileUpload(video, chunksize=4 << 20, resumable=True))
            resp, fails = None, 0
            while resp is None:
                try:
                    _, resp = req.next_chunk(num_retries=5)
                except HttpError as e:
                    if e.resp.status in (500, 502, 503, 504) and fails < 8:
                        fails += 1; print("upload server error, retry", fails); time.sleep(5 * fails); continue
                    raise
                except (OSError, ssl.SSLError) as e:  # connection dropped mid-upload: resume
                    fails += 1
                    print("upload network error, resuming", fails, e)
                    if fails > 12:
                        raise
                    time.sleep(5 * fails)
            break
        except HttpError as e:
            if attempt == 0 and "containsSyntheticMedia" in str(e):
                status.pop("containsSyntheticMedia"); continue
            raise
    try:
        yt.thumbnails().set(videoId=resp["id"], media_body=MediaFileUpload(thumb)).execute()
    except Exception as e:
        print("Thumbnail not set:", e)
    return resp["id"]


# ---------------------------------------------------------------- main
def _hms(t):
    t = int(t)
    return f"{t // 3600}:{t % 3600 // 60:02d}:{t % 60:02d}" if t >= 3600 else f"{t // 60}:{t % 60:02d}"


def final_description(s, times):
    """description + auto chapters (needs 0:00 first, >=3 chapters, >=10 s apart) + hashtags."""
    parts = [s["description"]]
    pts = {}
    for c in s.get("chapters") or []:
        try:
            i, title = int(c["scene"]) - 1, str(c["title"]).strip()
        except Exception:
            continue
        if 0 <= i < len(times) and title:
            pts[round(times[i][0], 1)] = title
    if pts:
        pts[0.0] = pts.get(0.0) or pts[min(pts)]
        last, lines = -99, []
        for t in sorted(pts):
            if t - last >= 10:
                lines.append(f"{_hms(t)} {pts[t]}"); last = t
        if len(lines) >= 3:
            parts.append("\n".join(lines))
    if s.get("_hashtags"):
        parts.append(s["_hashtags"])
    return "\n\n".join(parts)


def make_one(yt, hist, stats, fmt, idx):
    print(f"=== Video {idx + 1}: {fmt} | best hour IST: {stats['best_hour']}")
    t = pick_topic(yt, stats, fmt, hist)
    print("Topic:", t)
    s = write_script(t, fmt)
    work = f"work_{fmt}"
    os.makedirs(work, exist_ok=True)
    imgs = make_images(s, fmt, work)
    times, caps = make_voice(s, work, 3 if fmt == "short" else 5, use_gemini=not QUICK)
    mix_audio(s, times, work)
    video = render(s, fmt, work, imgs, times, caps)
    thumb_src = make_thumb_image(s, work)
    thumb = make_thumb(os.path.join(work, thumb_src or imgs[0]), s.get("thumb_text") or s["title"], os.path.join(work, "thumb.jpg"))
    s["description"] = final_description(s, times)
    if QUICK:
        s["title"] = "[TEST] " + s["title"][:85]
    slot = None if (REVIEW or QUICK) else next_slot(stats["best_hour"], 3 * idx)
    vid = upload(yt, video, thumb, s, slot)
    print("Uploaded:", vid, "scheduled:", slot)
    hist.append(dict(date=dt.datetime.now(IST).date().isoformat(), topic=t["topic"], title=s["title"], fmt=fmt, id=vid, when=slot or "private"))
    json.dump(hist, open("history.json", "w"), ensure_ascii=False, indent=1)
    write_dashboard(stats, hist)


def main():
    yt = youtube()
    hist = json.load(open("history.json")) if os.path.exists("history.json") else []
    today = dt.datetime.now(IST).date().isoformat()
    if os.getenv("GITHUB_EVENT_NAME") == "schedule" and any(h.get("date") == today for h in hist):
        print("Aaj ka video ban chuka hai, scheduled run skip.")
        return
    stats = analyze(yt)
    write_dashboard(stats, hist)
    if cf_configured() and not ALLOW_FALLBACK:
        cloudflare_image("a red apple on a table", steps=1)  # tiny check: is any account's free quota left?
        if not cf_available():
            print("Koi bhi Cloudflare account use nahi ho sakta (limit khatam ya token galat). Video nahi banega.")
            print("Limit roz 5:30 AM IST pe reset hoti hai.")
            sys.exit(1)
    if QUICK:
        plan = ["short"]  # quick test: one small Short, private
        print("QUICK TEST MODE: 6 scenes, edge voice, private upload")
    elif (os.getenv("RUN_BOTH") or "").lower() == "true" or os.path.exists("run_both.txt"):
        plan = ["short", "long"]  # test mode: short first (quick check), then long
    else:
        plan = [os.getenv("FORCE_FORMAT") or choose_format(stats)]
    errors = 0
    for i, fmt in enumerate(plan):
        try:
            make_one(yt, hist, stats, fmt, i)
        except Exception:
            errors += 1
            traceback.print_exc()
    if not os.path.exists("history.json"):
        json.dump(hist, open("history.json", "w"))
    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
