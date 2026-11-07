"""yt-autopilot v2: trending topic -> story script -> Gemini images (consistent characters)
-> multi-voice TTS -> music + sfx -> Ken Burns video -> thumbnail -> scheduled upload -> dashboard data."""
import os, io, re, sys, glob, math, json, time, base64, random, wave, asyncio, subprocess, traceback, datetime as dt
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
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
SR = 24000
GEM = "https://generativelanguage.googleapis.com/v1beta/models/"


# ---------------------------------------------------------------- helpers
def run(cmd, cwd=None):
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-1500:])


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


TEXT_PREF = ["gemini-3.8-flash", "gemini-3.6-flash", "gemini-3.5-flash", "gemini-3.5-flash-lite",
             "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-flash-latest",
             "gemini-2.5-flash", "gemini-2.5-flash-lite"]
IMG_PREF = ["gemini-3.1-flash-image", "gemini-3.1-flash-image-preview", "gemini-2.5-flash-image", "gemini-3-pro-image"]


def text_models():
    av = available_models()
    c = [os.getenv("GEMINI_MODEL")] if os.getenv("GEMINI_MODEL") else []
    c += [m for m in TEXT_PREF if m in av or not av]
    c += [m for m in av if re.fullmatch(r"gemini-[\d.]+-flash(-lite)?", m) and m not in c]
    return c


def gemini_json(prompt, temperature=0.8):
    body = {"contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"responseMimeType": "application/json", "temperature": temperature}}
    last = None
    for model in text_models():
        for i in range(2):
            try:
                r = requests.post(GEM + model + ":generateContent", params={"key": os.environ["GEMINI_API_KEY"]},
                                  json=body, timeout=180)
                if r.status_code in (400, 403, 404, 429):
                    last = f"{model}: {r.status_code} {r.text[:150]}"
                    print("text model skipped ->", last)
                    break
                r.raise_for_status()
                parts = r.json()["candidates"][0]["content"]["parts"]
                txt = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                print("text model used:", model)
                return json.loads(txt)
            except Exception as e:
                last = f"{model}: {e}"
                time.sleep(6)
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
                r = requests.post(GEM + model + ":generateContent", params={"key": os.environ["GEMINI_API_KEY"]},
                                  json={"contents": [{"role": "user", "parts": parts}], "generationConfig": cfg}, timeout=180)
            except Exception as e:
                print("img error", e); time.sleep(10); continue
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


CF_DEAD = False


def cloudflare_image(prompt, w, h):
    """Free: Cloudflare Workers AI FLUX-1-schnell (10,000 neurons/day)."""
    global CF_DEAD
    acc, tok = os.getenv("CF_ACCOUNT_ID"), os.getenv("CF_API_TOKEN")
    if CF_DEAD or not (acc and tok):
        return None
    url = f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/@cf/black-forest-labs/flux-1-schnell"
    for attempt in range(3):
        body = {"prompt": prompt[:1900], "steps": 6}
        if attempt == 0:
            body.update(width=w, height=h)  # some models reject custom size -> retried without
        try:
            r = requests.post(url, headers={"Authorization": f"Bearer {tok}"}, json=body, timeout=150)
        except Exception as e:
            print("cloudflare error:", e); time.sleep(8); continue
        if r.status_code == 200:
            b64 = (r.json().get("result") or {}).get("image")
            if b64:
                return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
            return None
        print("cloudflare", r.status_code, r.text[:200])
        if r.status_code in (401, 403):
            CF_DEAD = True
            return None
        if r.status_code == 429:
            time.sleep(20)
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
    n = 12 if short else 24
    prompt = f"""Write a high-retention illustrated {'Short' if short else 'video'} for YouTube. Language of speech/title/description: {LANG}
(Hindi = Devanagari, simple spoken words). Topic: {t['topic']}. Angle: {t['angle']}.
{'Total spoken words 110-130, hook in first 2 seconds.' if short else 'Total spoken words 800-950, hook in first 10 seconds, clear payoff, short CTA at end.'}
The VERY FIRST spoken line must be a hook under 12 words (shocking fact, bold claim or question), never a greeting like 'namaste dosto'. Accurate facts only, no clickbait lies, short punchy lines, a surprising fact or question every few seconds, end each scene on a mini-cliffhanger so viewers keep watching. Use 1-3 recurring characters ONLY if they make it better
(e.g. a curious kid + a wise teacher); narrator speaks the rest.
Make exactly {n} scenes. Each scene: "image_prompt" (English, vivid, describes action/setting/camera, NO text in image),
"characters" (ids appearing), "sfx" (one of none, whoosh, impact, ding), "lines" (1-3 items {{"speaker": "narrator" or character id, "text": "..."}}).
Characters: {{"id":"c1","name":"","look":"detailed fixed appearance: age, clothes, colors, hair","voice":"male|female|child|old"}}.
"style": ONE art style sentence reused for every image (e.g. "vibrant 3D animated film style, warm cinematic lighting").
"music_mood": one of calm, happy, sad, tense, epic.
Also "title" (max 70 chars, honest curiosity), "description" (3-5 lines + hashtags), "tags" (10-15), "thumb_text" (max 4 words in {LANG}).
Return JSON with keys: title, description, tags, thumb_text, style, music_mood, characters, scenes."""
    s = gemini_json(prompt)
    s = proofread(s)
    s["title"] = s["title"][:95]
    if short:
        s["description"] += "\n#Shorts"
    for sc in s["scenes"]:
        sc["lines"] = [l for l in sc.get("lines", []) if l.get("text")] or [{"speaker": "narrator", "text": "..."}]
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
    if not (os.getenv("CF_ACCOUNT_ID") and os.getenv("CF_API_TOKEN")):
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
        if img is None:
            img, src = pollinations_image(prompt, gw, gh), "pollinations"
        print(f"scene {i + 1}/{n}: {'OK via ' + src if img else 'FAILED'}")
        imgs[i] = img
    good = [i for i, im in enumerate(imgs) if im is not None]
    if not good:
        raise RuntimeError("Koi image nahi ban payi. Cloudflare keys (CF_ACCOUNT_ID, CF_API_TOKEN) check karo.")
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
            wait = 21 - (time.time() - _TTS_LAST[0])
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
                r = requests.post(GEM + model + ":generateContent", params={"key": os.environ["GEMINI_API_KEY"]},
                                  json=body, timeout=180)
            except Exception as e:
                print("tts error", e); continue
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


def sfx(kind):
    if kind == "whoosh":
        n = np.convolve(np.random.randn(int(0.7 * SR)), np.ones(40) / 40, "same") * 3
        return n * np.sin(np.linspace(0, np.pi, len(n))) ** 2
    x = np.arange(int(1.0 * SR)) / SR
    if kind == "impact":
        return np.sin(2 * np.pi * 55 * x) * np.exp(-x * 6) + 0.3 * np.random.randn(len(x)) * np.exp(-x * 30)
    if kind == "ding":
        return (np.sin(2 * np.pi * 1320 * x) + 0.5 * np.sin(2 * np.pi * 1980 * x)) * np.exp(-x * 4) * 0.5
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


def user_music(mood, total, work):
    """Real tracks the owner uploaded to the music/ folder (name them like calm_1.mp3, happy_2.mp3 ...)."""
    files = [f for f in glob.glob("music/*") if f.lower().endswith((".mp3", ".wav", ".m4a", ".ogg", ".aac"))]
    if not files:
        return None
    pick = [f for f in files if mood in os.path.basename(f).lower()] or files
    f = random.choice(pick)
    out = os.path.join(work, "music.wav")
    try:
        run(["ffmpeg", "-y", "-stream_loop", "-1", "-i", os.path.abspath(f), "-t", f"{total:.2f}", "-ar", str(SR),
             "-ac", "1", "-sample_fmt", "s16", "-af", f"afade=t=in:d=2,afade=t=out:st={max(0, total - 3):.2f}:d=3", out])
        x = read_wav(out)
        print("music track:", os.path.basename(f))
        return x / max(1e-6, np.abs(x).max())
    except Exception as e:
        print("music file failed:", e)
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
    real = track is not None
    if track is None:
        try:
            track = lofi(mood, total)
            print("music: generated", mood, "lo-fi track")
        except Exception as e:
            print("lofi failed, using simple pad:", e)
            track = pad(mood, total)
    track = np.pad(track, (0, max(0, len(voice) - len(track))))[:len(voice)]
    bed = track * (0.20 if real else 0.17) * duck
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
    mix = voice + bed + fx * 0.30
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


def _font(size):
    try:
        f = subprocess.run(["fc-match", "-f", "%{file}", "Noto Sans Devanagari:bold"], capture_output=True, text=True).stdout
        return ImageFont.truetype(f, size)
    except Exception:
        return ImageFont.load_default()


def make_thumb(img_path, text, out):
    img = Image.open(img_path).convert("RGB")
    r = max(1280 / img.width, 720 / img.height)
    img = img.resize((int(img.width * r), int(img.height * r)))
    l, t = (img.width - 1280) // 2, (img.height - 720) // 2
    img = ImageEnhance.Brightness(ImageEnhance.Color(img.crop((l, t, l + 1280, t + 720))).enhance(1.3)).enhance(0.7)
    d = ImageDraw.Draw(img)
    lines, cur = [], ""
    for w in text.split():
        if len(cur) + len(w) > 14 and cur:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    size = 150 if len(lines) <= 2 else 115
    f, y = _font(size), (720 - len(lines) * (size + 20)) // 2
    for ln in lines:
        d.text(((1280 - d.textlength(ln, font=f)) / 2, y), ln, font=f, fill="#FFE600", stroke_width=10, stroke_fill="black")
        y += size + 20
    img.save(out, "JPEG", quality=90)
    return out


# ---------------------------------------------------------------- upload
def upload(yt, video, thumb, s, publish_at):
    status = {"privacyStatus": "private", "selfDeclaredMadeForKids": False, "containsSyntheticMedia": True}
    if publish_at:
        status["publishAt"] = publish_at
    body = {"snippet": {"title": s["title"], "description": s["description"], "tags": s["tags"][:15],
                        "categoryId": "27", "defaultLanguage": LANG, "defaultAudioLanguage": LANG}, "status": status}
    for attempt in (0, 1):
        try:
            req = yt.videos().insert(part="snippet,status", body=body,
                                     media_body=MediaFileUpload(video, chunksize=8 << 20, resumable=True))
            resp = None
            while resp is None:
                _, resp = req.next_chunk()
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
def make_one(yt, hist, stats, fmt, idx):
    print(f"=== Video {idx + 1}: {fmt} | best hour IST: {stats['best_hour']}")
    t = pick_topic(yt, stats, fmt, hist)
    print("Topic:", t)
    s = write_script(t, fmt)
    work = f"work_{fmt}"
    os.makedirs(work, exist_ok=True)
    imgs = make_images(s, fmt, work)
    times, caps = make_voice(s, work, 3 if fmt == "short" else 5)
    mix_audio(s, times, work)
    video = render(s, fmt, work, imgs, times, caps)
    thumb = make_thumb(os.path.join(work, imgs[0]), s["thumb_text"], os.path.join(work, "thumb.jpg"))
    slot = None if REVIEW else next_slot(stats["best_hour"], 3 * idx)
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
    if (os.getenv("RUN_BOTH") or "").lower() == "true" or os.path.exists("run_both.txt"):
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
