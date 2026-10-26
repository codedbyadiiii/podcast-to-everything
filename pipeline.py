"""
Podcast-to-Everything pipeline (Group 05)
Audio -> Whisper -> clean transcript -> LLM (summary, key points, X/LinkedIn/Instagram)
      -> translation -> TTS (translated MP3) -> faithfulness check
"""
import os, re, json, zipfile

# ----------------------------------------------------------------- LLM backends
class LLM:
    """provider='openai' (best quality, needs key) or 'local' (free, no key, uses Qwen on Colab GPU)."""

    def __init__(self, provider=None, openai_model="gpt-4o-mini",
                 local_model="Qwen/Qwen2.5-1.5B-Instruct"):
        if provider is None:
            provider = "openai" if os.environ.get("OPENAI_API_KEY") else "local"
        self.provider = provider
        if provider == "openai":
            from openai import OpenAI
            self.model_name = openai_model
            self.client = OpenAI()
            self.max_chars = 12000        # chunk size for long transcripts
        else:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.model_name = local_model
            self.tok = AutoTokenizer.from_pretrained(local_model)
            self.model = AutoModelForCausalLM.from_pretrained(local_model)
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            if self.device == "cuda":
                self.model = self.model.half()
            self.model.to(self.device)
            self.max_chars = 3500
        print(f"LLM ready -> {self.provider}: {self.model_name}")

    def __call__(self, system, user, max_tokens=700, temperature=0.2):
        if self.provider == "openai":
            r = self.client.chat.completions.create(
                model=self.model_name, temperature=temperature, max_tokens=max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
            return r.choices[0].message.content.strip()
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        text = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = self.tok(text, return_tensors="pt").to(self.model.device)
        kw = dict(max_new_tokens=max_tokens, do_sample=temperature > 0,
                  pad_token_id=self.tok.eos_token_id)
        if temperature > 0:
            kw["temperature"] = temperature
        out = self.model.generate(**inp, **kw)
        return self.tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True).strip()


# ------------------------------------------------------------- speech to text
def load_whisper(size="base"):
    import whisper
    return whisper.load_model(size)

def transcribe(model, audio_path, language=None):
    import torch
    res = model.transcribe(audio_path, fp16=torch.cuda.is_available(), language=language)
    return res["text"].strip(), res.get("language", "en")


# ------------------------------------------------------------ text preparation
def split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]

def clean_transcript(raw, sentences_per_para=6):
    """Rule-based cleanup (no LLM, so nothing can be hallucinated): fix spacing, remove
    immediate word repeats ('the the'), and break the wall of text into paragraphs."""
    t = re.sub(r"\s+", " ", raw).strip()
    t = re.sub(r"\b(\w+)(\s+\1\b)+", r"\1", t, flags=re.I)
    sents = split_sentences(t)
    paras = [" ".join(sents[i:i + sentences_per_para]) for i in range(0, len(sents), sentences_per_para)]
    return "\n\n".join(paras)

def chunk_text(text, max_chars):
    chunks, cur = [], ""
    for s in split_sentences(text.replace("\n", " ")):
        if len(cur) + len(s) + 1 > max_chars and cur:
            chunks.append(cur.strip()); cur = ""
        cur += " " + s
    if cur.strip():
        chunks.append(cur.strip())
    return chunks or [text]


# ------------------------------------------------------------- LLM generation
GROUND = ("You are a careful assistant that works ONLY from the provided podcast transcript. "
          "Never add facts, numbers, names or claims that are not in the transcript. "
          "Keep the speaker's opinions as opinions (e.g. 'the speaker says...'). "
          "Preserve important numbers exactly as stated.")

def condense(llm, transcript):
    """Map step for long podcasts: faithful notes per chunk. Short podcasts are used as-is."""
    chunks = chunk_text(transcript, llm.max_chars)
    if len(chunks) == 1:
        return transcript
    notes = []
    for i, c in enumerate(chunks, 1):
        print(f"  condensing part {i}/{len(chunks)}")
        notes.append(llm(GROUND, "Write faithful bullet-point notes (max 10 bullets) of this part "
                                 "of a podcast transcript.\n\n" + c, max_tokens=450, temperature=0))
    return "\n".join(notes)

def make_summary(llm, source):
    return llm(GROUND, "Write a concise summary of this podcast in 120-180 words, one or two "
                       "paragraphs of plain prose.\n\nTranscript:\n" + source, max_tokens=420, temperature=0.2)

def make_key_points(llm, source):
    return llm(GROUND, "List the 5-8 most important key points of this podcast as short bullet "
                       "points, each starting with '- '. Output only the bullets.\n\nTranscript:\n" + source,
               max_tokens=450, temperature=0.2)

def _fit_x(llm, post, limit=280):
    if len(post) <= limit:
        return post
    post = llm(GROUND, f"Rewrite this post to be under {limit} characters, keep the meaning:\n\n{post}",
               max_tokens=150, temperature=0)
    if len(post) > limit:
        post = post[:limit - 1].rsplit(" ", 1)[0] + "…"
    return post

def make_social_posts(llm, source):
    x = llm(GROUND, "Write ONE engaging post for X (Twitter), strictly under 280 characters, with "
                    "at most 2 hashtags. Output only the post.\n\nTranscript:\n" + source,
            max_tokens=120, temperature=0.5)
    li = llm(GROUND, "Write ONE LinkedIn post (110-170 words): a strong first line, 3 short "
                     "paragraphs, then 3-5 relevant hashtags. Professional tone. Output only the "
                     "post.\n\nTranscript:\n" + source, max_tokens=350, temperature=0.5)
    ig = llm(GROUND, "Write ONE Instagram caption: friendly, 2-4 short lines with a few emojis, "
                     "then 6-10 hashtags. Output only the caption.\n\nTranscript:\n" + source,
             max_tokens=250, temperature=0.6)
    return {"x": _fit_x(llm, x.strip().strip('"')), "linkedin": li, "instagram": ig}


# -------------------------------------------------------- translation and TTS
LANG_NAMES = {"hi": "Hindi", "mr": "Marathi", "en": "English", "gu": "Gujarati", "ta": "Tamil",
              "te": "Telugu", "bn": "Bengali", "kn": "Kannada", "ur": "Urdu", "fr": "French",
              "es": "Spanish", "de": "German"}

def translate(text, target="hi", source="auto", llm=None, engine="google"):
    """engine='google' (free, deterministic, default) or 'llm'."""
    if engine == "llm" and llm is not None:
        parts = [llm("You are a professional translator. Output only the translation.",
                     f"Translate to {LANG_NAMES.get(target, target)}, keep meaning, numbers and "
                     f"bullet markers unchanged:\n\n{c}", max_tokens=1500, temperature=0)
                 for c in chunk_text(text, 3000)]
        return "\n".join(parts)
    from deep_translator import GoogleTranslator
    tr = GoogleTranslator(source=source, target=target)
    out = []
    for line in text.split("\n"):            # keep line structure (bullets, paragraphs)
        if not line.strip():
            out.append(""); continue
        pieces = [tr.translate(c) for c in chunk_text(line, 4000)]
        out.append(" ".join(p for p in pieces if p))
    return "\n".join(out)

def text_to_speech(text, lang, path):
    from gtts import gTTS
    clean = re.sub(r"[#*_`>-]+", " ", text)          # remove markdown symbols so they are not read out
    clean = re.sub(r"\s+", " ", clean).strip()
    gTTS(text=clean, lang=lang, slow=False).save(path)
    return path


# ----------------------------------------------------------- faithfulness check
STOP = set("""the and for that this with from have has had are was were been being will would could should
about into over under than then them they their there what when where which while who whom whose your
you our out not but also just very more most some such only like even much many make made does did can
its it's is a an of to in on at by or as be we i he she him her his hers us""".split())

def _words(text):
    return [re.sub(r"(ing|ed|es|s)$", "", w) for w in re.findall(r"[a-z']+", text.lower())
            if len(w) > 3 and w not in STOP]

def _numbers(text):
    return set(n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)?", text))

def _claims(output):
    claims = []
    for line in output.split("\n"):
        line = re.sub(r"^[\s\-\*•\d.)]+", "", line).strip()
        claims += [s for s in split_sentences(line) if len(s.split()) >= 4]
    claims = [re.sub(r"#\w+", "", c).strip() for c in claims]
    return [c for c in claims if len(c.split()) >= 4][:25]

def lexical_support(output, source, threshold=0.6):
    vocab = set(_words(source))
    claims = _claims(output)
    if not claims:
        return 1.0, []
    weak = []
    for c in claims:
        w = _words(c)
        cov = sum(x in vocab for x in w) / max(len(w), 1)
        if cov < threshold:
            weak.append(c)
    return round(1 - len(weak) / len(claims), 2), weak

def llm_judge(llm, output, source):
    claims = _claims(output)
    if not claims:
        return None, []
    numbered = "\n".join(f"{i}. {c}" for i, c in enumerate(claims, 1))
    ans = llm("You are a strict fact checker.",
              "Transcript:\n" + source[:24000] + "\n\nFor each numbered claim below, answer YES if the "
              "transcript supports it (paraphrase is fine) or NO if it adds or changes information. "
              "Reply with one line per claim in the form '<number>: YES' or '<number>: NO'.\n\n" + numbered,
              max_tokens=250, temperature=0)
    verdict = {int(n): v for n, v in re.findall(r"(?m)^\s*(\d+)\s*[:.)-]\s*(YES|NO)", ans.upper())}
    if not verdict:
        return None, []
    bad = [claims[i - 1] for i, v in verdict.items() if v == "NO" and i <= len(claims)]
    return round(1 - len(bad) / len(verdict), 2), bad

def faithfulness_report(llm, outputs, source):
    """outputs: {name: text}. Checks each output against the transcript (or notes for long podcasts)."""
    rep = {}
    src_nums = _numbers(source)
    for name, text in outputs.items():
        lex, weak = lexical_support(text, source)
        judge, bad = llm_judge(llm, text, source)
        missing = sorted(_numbers(text) - src_nums)
        scores = [s for s in (lex, judge) if s is not None]
        rep[name] = {"lexical_support": lex, "llm_judge": judge,
                     "numbers_not_in_transcript": missing,
                     "flagged_claims": list(dict.fromkeys(weak + bad)),
                     "status": "PASS" if min(scores) >= 0.8 and not missing else "REVIEW"}
    return rep

def back_translation_score(translated, original, lang):
    """Translate back to English and measure how much of the original's vocabulary survives."""
    from deep_translator import GoogleTranslator
    back = GoogleTranslator(source=lang, target="en").translate(translated[:4500])
    o, b = set(_words(original)), set(_words(back))
    return round(len(o & b) / max(len(o), 1), 2), back


# ------------------------------------------------------------------ full run
def run_pipeline(audio_path, target_langs=("hi",), whisper_size="base", llm=None, out_dir="outputs",
                 translate_engine="google", whisper_model=None, check=True, make_audio=True):
    os.makedirs(out_dir, exist_ok=True)
    llm = llm or LLM()
    print("1/6 Transcribing...")
    wm = whisper_model or load_whisper(whisper_size)
    raw, detected = transcribe(wm, audio_path)
    transcript = clean_transcript(raw)
    print(f"    detected language: {detected}, {len(transcript.split())} words")

    print("2/6 Understanding the podcast...")
    source = condense(llm, transcript)

    print("3/6 Summary, key points, social posts...")
    summary = make_summary(llm, source)
    key_points = make_key_points(llm, source)
    social = make_social_posts(llm, source)

    res = {"language": detected, "transcript": transcript, "summary": summary,
           "key_points": key_points, "social": social, "translations": {}, "audio": {}}

    print("4/6 Translating...")
    for lg in target_langs:
        res["translations"][lg] = {
            "summary": translate(summary, lg, llm=llm, engine=translate_engine),
            "key_points": translate(key_points, lg, llm=llm, engine=translate_engine)}

    if make_audio:
        print("5/6 Text-to-speech...")
        for lg in target_langs:
            t = res["translations"][lg]
            body = t["summary"] + "\n" + t["key_points"]
            res["audio"][lg] = text_to_speech(body, lg, os.path.join(out_dir, f"podcast_summary_{lg}.mp3"))

    if check:
        print("6/6 Faithfulness check...")
        res["faithfulness"] = faithfulness_report(
            llm, {"summary": summary, "key_points": key_points, **{f"post_{k}": v for k, v in social.items()}},
            source)
        res["back_translation"] = {}
        for lg in target_langs:
            try:
                s, _ = back_translation_score(res["translations"][lg]["summary"], summary, lg)
                res["back_translation"][lg] = s
            except Exception as e:
                res["back_translation"][lg] = f"skipped ({e.__class__.__name__})"
    save_outputs(res, out_dir)
    return res

def save_outputs(res, out_dir="outputs"):
    w = lambda n, t: open(os.path.join(out_dir, n), "w", encoding="utf-8").write(t)
    w("transcript.txt", res["transcript"])
    w("summary.md", res["summary"])
    w("key_points.md", res["key_points"])
    s = res["social"]
    w("social_posts.md", f"## X\n{s['x']}\n\n## LinkedIn\n{s['linkedin']}\n\n## Instagram\n{s['instagram']}\n")
    for lg, t in res["translations"].items():
        w(f"summary_{lg}.md", t["summary"]); w(f"key_points_{lg}.md", t["key_points"])
    if "faithfulness" in res:
        w("faithfulness_report.json", json.dumps(
            {"faithfulness": res["faithfulness"], "back_translation": res.get("back_translation")},
            indent=2, ensure_ascii=False))
    zp = out_dir.rstrip("/") + ".zip"
    with zipfile.ZipFile(zp, "w") as z:
        for f in os.listdir(out_dir):
            z.write(os.path.join(out_dir, f), f)
    res["zip"] = zp
