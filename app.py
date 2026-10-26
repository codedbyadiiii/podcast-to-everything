import os
import tempfile

import streamlit as st
import pipeline as p

st.set_page_config(page_title="Podcast-to-Everything", page_icon="🎙️", layout="wide")


def configured_secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:
        return ""


def get_llm(api_key):
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        return p.LLM("openai")
    return p.LLM("local")


@st.cache_resource
def get_cached_llm(api_key):
    return get_llm(api_key)


@st.cache_resource
def get_whisper(size):
    return p.load_whisper(size)


st.title("Podcast-to-Everything")
st.caption("One upload becomes a transcript, summary, social content, translations, audio, and a faithfulness report.")

with st.sidebar:
    st.subheader("Pipeline settings")
    configured_key = configured_secret("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY", "")
    if configured_key:
        st.success("OpenAI is configured")
        key = configured_key
    else:
        key = st.text_input("OpenAI API key", type="password", help="For deployment, add this as a Streamlit secret named OPENAI_API_KEY.")
        st.info("Without a key, the app downloads a local 1.5B model and may be slow or exceed cloud memory limits.")

    size = st.selectbox("Whisper model", ["tiny", "base", "small"], index=0)
    langs = st.multiselect("Translate to", ["hi", "mr", "gu", "ta", "te", "bn", "kn", "ur", "fr", "es", "de"],
                           default=["hi", "mr"], format_func=lambda c: p.LANG_NAMES.get(c, c))
    check = st.checkbox("Run faithfulness check", True)
    make_audio = st.checkbox("Generate translated audio", True)

up = st.file_uploader("Upload podcast audio", type=["mp3", "wav", "m4a", "ogg", "flac"])
run = st.button("Run pipeline", type="primary", disabled=not up or not langs)
if run:
    run_dir = tempfile.mkdtemp(prefix="podcast-to-everything-")
    path = os.path.join(run_dir, up.name)
    with open(path, "wb") as audio_file:
        audio_file.write(up.getvalue())

    try:
        with st.status("Running the podcast pipeline...", expanded=True) as status:
            st.write("Transcribing audio")
            st.write("Generating content and translations")
            if make_audio:
                st.write("Creating translated audio")
            if check:
                st.write("Checking faithfulness")
            st.session_state["res"] = p.run_pipeline(
                path,
                target_langs=tuple(langs),
                llm=get_cached_llm(key),
                whisper_model=get_whisper(size),
                translate_engine="llm" if key else "google",
                out_dir=os.path.join(run_dir, "outputs"),
                check=check,
                make_audio=make_audio,
            )
            status.update(label="Pipeline complete", state="complete", expanded=False)
    except Exception as exc:
        st.error(f"The pipeline failed: {exc}")
        st.stop()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

res = st.session_state.get("res")
if res:
    L = lambda c: p.LANG_NAMES.get(c, c)
    overview, transcript_tab, social_tab, translation_tab = st.tabs(
        ["Overview", "Transcript", "Social posts", "Translations"]
    )
    with overview:
        st.subheader("Summary")
        st.write(res["summary"])
        st.subheader("Key points")
        st.markdown(res["key_points"])
    with transcript_tab:
        st.text_area("Transcript", res["transcript"], height=420)
    with social_tab:
        for k, title in [("x", "X / Twitter"), ("linkedin", "LinkedIn"), ("instagram", "Instagram")]:
            st.subheader(title)
            st.write(res["social"][k])
    with translation_tab:
        for lg, tr in res["translations"].items():
            st.subheader(f"{L(lg)} summary")
            st.write(tr["summary"])
            st.subheader(f"{L(lg)} key points")
            st.markdown(tr["key_points"])
            if lg in res["audio"]:
                st.audio(res["audio"][lg], format="audio/mp3")
    if "faithfulness" in res:
        with st.expander("Faithfulness check"):
            st.json(res["faithfulness"]); st.write({"back_translation": res.get("back_translation")})
    with open(res["zip"], "rb") as z:
        st.download_button("Download all outputs (.zip)", z, "podcast_outputs.zip")
