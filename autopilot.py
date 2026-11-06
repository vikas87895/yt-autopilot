"""yt-autopilot v2: trending topic -> story script -> Gemini images (consistent characters)
-> multi-voice TTS -> music + sfx -> Ken Burns video -> thumbnail -> scheduled upload -> dashboard data."""
import os, io, re, sys, json, time, base64, random, wave, asyncio, subprocess, traceback, datetime as dt
import requests, numpy as np
import edge_tts
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from googleapiclient.errors import HttpError

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


def fallback_image(prompt, w, h, i):
    try:
        r = requests.get("https://image.pollinations.ai/prompt/" + requests.utils.quote(prompt[:400]),
                         params={"width": w, "height": h, "nologo": "true", "seed": random.randint(1, 99999)}, timeout=120)
        if r.ok and r.headers.get("content-type", "").startswith("image"):
            return Image.open(io.BytesIO(r.content)).convert("RGB")
    except Exception:
        pass
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    c1, c2 = [(30, 20, 60), (10, 50, 80), (60, 20, 40), (15, 60, 50)][i % 4], (5, 5, 15)
    for y in range(h):
        k = y / h
        d.line([(0, y), (w, y)], fill=tuple(int(c1[j] * (1 - k) + c2[j] * k) for j in range(3)))
    return img


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


def write_script(t, fmt):
    short = fmt == "short"
    n = 8 if short else 16
    prompt = f"""Write a high-retention illustrated {'Short' if short else 'video'} for YouTube. Language of speech/title/description: {LANG}
(Hindi = Devanagari, simple spoken words). Topic: {t['topic']}. Angle: {t['angle']}.
{'Total spoken words 110-130, hook in first 2 seconds.' if short else 'Total spoken words 800-950, hook in first 10 seconds, clear payoff, short CTA at end.'}
Accurate facts only, no clickbait lies, short punchy lines. Use 1-3 recurring characters ONLY if they make it better
(e.g. a curious kid + a wise teacher); narrator speaks the rest.
Make exactly {n} scenes. Each scene: "image_prompt" (English, vivid, describes action/setting/camera, NO text in image),
"characters" (ids appearing), "sfx" (one of none, whoosh, impact, ding), "lines" (1-3 items {{"speaker": "narrator" or character id, "text": "..."}}).
Characters: {{"id":"c1","name":"","look":"detailed fixed appearance: age, clothes, colors, hair","voice":"male|female|child|old"}}.
"style": ONE art style sentence reused for every image (e.g. "vibrant 3D animated film style, warm cinematic lighting").
"music_mood": one of calm, happy, sad, tense, epic.
Also "title" (max 70 chars, honest curiosity), "description" (3-5 lines + hashtags), "tags" (10-15), "thumb_text" (max 4 words in {LANG}).
Return JSON with keys: title, description, tags, thumb_text, style, music_mood, characters, scenes."""
    s = gemini_json(prompt)
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
    aspect = "9:16" if short else "16:9"
    style, refs = s["style"], {}
    for c in s.get("characters", [])[:4]:
        r = gemini_image(f"{style}. Character reference portrait of {c['name']}: {c['look']}. "
                         "Full body, plain neutral background, no text, no watermark.", [], "1:1")
        if r:
            refs[c["id"]] = r
        time.sleep(4)
    paths = []
    for i, sc in enumerate(s["scenes"]):
        rf = [refs[c] for c in sc.get("characters", []) if c in refs][:3]
        prompt = (f"{style}. {sc['image_prompt']}. " + ("Characters must look exactly like the reference images. " if rf else "")
                  + "No text, no captions, no watermark, no logo.")
        r = gemini_image(prompt, rf, aspect)
        time.sleep(4)
        img = Image.open(io.BytesIO(base64.b64decode(r[1]))).convert("RGB") if r else fallback_image(prompt, W, H, i)
        p = f"img{i}.jpg"
        img.save(os.path.join(work, p), quality=92)
        paths.append(p)
    return paths


# ---------------------------------------------------------------- voices
VOICES = {
    "hi": {"narrator": ("hi-IN-MadhurNeural", "+0Hz", "+0%"), "male": ("hi-IN-MadhurNeural", "-8Hz", "+3%"),
           "female": ("hi-IN-SwaraNeural", "+0Hz", "+0%"), "child": ("hi-IN-SwaraNeural", "+55Hz", "+12%"),
           "old": ("hi-IN-MadhurNeural", "-25Hz", "-12%")},
    "en": {"narrator": ("en-IN-PrabhatNeural", "+0Hz", "+0%"), "male": ("en-IN-PrabhatNeural", "-8Hz", "+3%"),
           "female": ("en-IN-NeerjaNeural", "+0Hz", "+0%"), "child": ("en-IN-NeerjaNeural", "+55Hz", "+12%"),
           "old": ("en-IN-PrabhatNeural", "-25Hz", "-12%")},
}


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


def make_voice(s, work, chunk):
    kinds = {c["id"]: c.get("voice", "male") for c in s.get("characters", [])}
    vmap = VOICES.get(LANG, VOICES["hi"])
    cursor, wavs, caps, times = 0.0, [], [], []
    for si, sc in enumerate(s["scenes"]):
        start = cursor
        for li, ln in enumerate(sc["lines"]):
            kind = "narrator" if ln["speaker"] == "narrator" else kinds.get(ln["speaker"], "male")
            mp3, wav = f"l{si}_{li}.mp3", f"l{si}_{li}.wav"
            words = asyncio.run(_tts(ln["text"], vmap.get(kind, vmap["narrator"]), os.path.join(work, mp3)))
            run(["ffmpeg", "-y", "-i", mp3, "-ar", str(SR), "-ac", "1", "-sample_fmt", "s16",
                 "-af", "apad=pad_dur=0.22", wav], work)
            d = dur(os.path.join(work, wav))
            caps += [(cursor + w[0], cursor + w[0] + w[1], w[2]) for w in words]
            cursor += d
            wavs.append(os.path.join(work, wav))
        times.append((start, cursor))
    with wave.open(os.path.join(work, "voice.wav"), "wb") as out:
        out.setnchannels(1); out.setsampwidth(2); out.setframerate(SR)
        for w in wavs:
            with wave.open(w) as r:
                out.writeframes(r.readframes(r.getnframes()))
    has = bool(caps)
    if has:
        with open(os.path.join(work, "caps.srt"), "w", encoding="utf-8") as f:
            for i in range(0, len(caps), chunk):
                g = caps[i:i + chunk]
                f.write(f"{i // chunk + 1}\n{_ts(g[0][0])} --> {_ts(g[-1][1])}\n{' '.join(x[2] for x in g)}\n\n")
    return times, has


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
    x = np.arange(int(total * SR)) / SR
    m = np.zeros_like(x)
    for i, f in enumerate(CHORDS.get(mood, CHORDS["calm"])):
        lfo = 0.6 + 0.4 * np.sin(2 * np.pi * (0.07 + 0.03 * i) * x + i)
        m += (np.sin(2 * np.pi * f * x) + 0.3 * np.sin(2 * np.pi * 2 * f * x)) * lfo
    m /= max(1e-6, np.abs(m).max())
    fade = np.minimum(1, np.minimum(x / 2, (total - x) / 2))
    return m * fade


def mix_audio(s, times, work):
    voice = read_wav(os.path.join(work, "voice.wav"))
    total = len(voice) / SR + 0.6
    voice = np.pad(voice, (0, int(total * SR) - len(voice)))
    k = int(0.25 * SR)
    cs = np.cumsum(np.insert(np.abs(voice), 0, 0))
    env = np.pad((cs[k:] - cs[:-k]) / k, (k // 2, len(voice) - (len(cs) - k) - k // 2))[:len(voice)]
    duck = 1 - 0.7 * np.clip(env * 12, 0, 1)
    bed = pad(s.get("music_mood", "calm"), total)[:len(voice)] * 0.16 * duck
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
    mix = voice + bed + fx * 0.35
    mix /= max(1.0, np.abs(mix).max() / 0.95)
    write_wav(os.path.join(work, "mix.wav"), mix)


# ---------------------------------------------------------------- video
def render(s, fmt, work, imgs, times, has_caps):
    short = fmt == "short"
    W, H = (720, 1280) if short else (1280, 720)
    parts = []
    for i, (img, (a, b)) in enumerate(zip(imgs, times)):
        d = b - a + (0.6 if i == len(times) - 1 else 0)
        fr = max(15, round(d * 30))
        mode = i % 3
        if mode == 0:
            z, x, y = "min(zoom+0.0007,1.2)", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        elif mode == 1:
            z, x, y = "if(eq(on,0),1.2,max(zoom-0.0007,1.0))", "iw/2-(iw/zoom/2)", "ih/2-(ih/zoom/2)"
        else:
            z, x, y = "1.15", f"(iw-iw/zoom)*on/{fr}", "ih/2-(ih/zoom/2)"
        vf = (f"scale={W * 2}:{H * 2}:force_original_aspect_ratio=increase,crop={W * 2}:{H * 2},"
              f"zoompan=z='{z}':x='{x}':y='{y}':d={fr}:s={W}x{H}:fps=30,setsar=1,format=yuv420p")
        out = f"clip{i}.mp4"
        run(["ffmpeg", "-y", "-loop", "1", "-i", img, "-vf", vf, "-frames:v", str(fr),
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", out], work)
        parts.append(out)
    with open(os.path.join(work, "list.txt"), "w") as f:
        f.writelines(f"file '{p}'\n" for p in parts)
    run(["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", "list.txt", "-c", "copy", "joined.mp4"], work)
    style = (f"FontName=Noto Sans Devanagari,Bold=1,FontSize={16 if short else 13},PrimaryColour=&H00FFFFFF&,"
             f"OutlineColour=&H00000000&,BorderStyle=1,Outline=2,Alignment=2,MarginV={110 if short else 30}")
    cmd = ["ffmpeg", "-y", "-i", "joined.mp4", "-i", "mix.wav"]
    if has_caps:
        cmd += ["-vf", f"subtitles=caps.srt:force_style='{style}'"]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-c:a", "aac", "-b:a", "160k",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", "final.mp4"]
    run(cmd, work)
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
    times, has_caps = make_voice(s, work, 3 if fmt == "short" else 5)
    mix_audio(s, times, work)
    video = render(s, fmt, work, imgs, times, has_caps)
    thumb = make_thumb(os.path.join(work, imgs[0]), s["thumb_text"], os.path.join(work, "thumb.jpg"))
    slot = None if REVIEW else next_slot(stats["best_hour"], 3 * idx)
    vid = upload(yt, video, thumb, s, slot)
    print("Uploaded:", vid, "scheduled:", slot)
    hist.append(dict(date=dt.date.today().isoformat(), topic=t["topic"], title=s["title"], fmt=fmt, id=vid, when=slot or "private"))
    json.dump(hist, open("history.json", "w"), ensure_ascii=False, indent=1)
    write_dashboard(stats, hist)


def main():
    yt = youtube()
    hist = json.load(open("history.json")) if os.path.exists("history.json") else []
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
