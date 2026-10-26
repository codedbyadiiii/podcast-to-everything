import os, tempfile, streamlit as st
import pipeline as p

st.set_page_config(page_title="Podcast-to-Everything", page_icon="🎙️", layout="wide")
st.title("🎙️ Podcast-to-Everything")
st.caption("Create once, repurpose everywhere — transcript, summary, key points, social posts, translation and audio.")

with st.sidebar:
    key = st.text_input("OpenAI API key (leave empty to use the free local model)", type="password",
                        value=os.environ.get("OPENAI_API_KEY", ""))
    size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=1)
    langs = st.multiselect("Translate to", ["hi", "mr", "gu", "ta", "te", "bn", "kn", "ur", "fr", "es", "de"],
                           default=["hi"], format_func=lambda c: p.LANG_NAMES.get(c, c))
    check = st.checkbox("Run faithfulness check", True)

@st.cache_resource
def get_llm(k):
    if k: os.environ["OPENAI_API_KEY"] = k
    return p.LLM("openai" if k else "local")

@st.cache_resource
def get_whisper(s): return p.load_whisper(s)

up = st.file_uploader("Upload podcast audio", type=["mp3", "wav", "m4a", "ogg", "flac"])
if up and st.button("Run pipeline", type="primary"):
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1]) as f:
        f.write(up.read()); path = f.name
    with st.spinner("Working... long podcasts can take a few minutes"):
        st.session_state["res"] = p.run_pipeline(path, tuple(langs), llm=get_llm(key),
                                                 whisper_model=get_whisper(size), check=check)

res = st.session_state.get("res")
if res:
    L = lambda c: p.LANG_NAMES.get(c, c)
    st.header("1️⃣ Transcribed text"); st.text_area("Transcript", res["transcript"], height=300)
    st.header("2️⃣ Summary"); st.write(res["summary"])
    st.header("3️⃣ Key points"); st.write(res["key_points"])
    st.header("4️⃣ Social media posts")
    for k, title in [("x", "X (Twitter)"), ("linkedin", "LinkedIn"), ("instagram", "Instagram")]:
        st.subheader(title); st.write(res["social"][k])
    st.header("5️⃣ Translated text")
    for lg, tr in res["translations"].items():
        st.subheader(L(lg)); st.write(tr["summary"]); st.write(tr["key_points"])
    st.header("6️⃣ Translated audio")
    for lg, path in res["audio"].items():
        st.subheader(L(lg)); st.audio(path)
    if "faithfulness" in res:
        with st.expander("Faithfulness check"):
            st.json(res["faithfulness"]); st.write({"back_translation": res.get("back_translation")})
    with open(res["zip"], "rb") as z:
        st.download_button("Download all outputs (.zip)", z, "podcast_outputs.zip")
