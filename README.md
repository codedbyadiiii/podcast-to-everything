# Podcast-to-Everything Pipeline

Convert one podcast into a transcript, summary, key points, social posts, translations, translated audio, and a faithfulness report.

## Project Partners

| Name | Student ID |
| --- | --- |
| Aryan Anil Dhondage | ADT23SOCB0201 |
| Aditya Nalge | ADT23SOCB0080 |
| Aditya Bade | ADT23SOCB0058 |

## Pipeline

`Audio -> Whisper -> Transcript -> LLM -> Summary / Social Posts -> Translation -> TTS`

## Features

- Local Whisper transcription
- Local Qwen model support without an API key
- Optional OpenAI support through `OPENAI_API_KEY`
- Hindi and Marathi translation support, with more languages available in `pipeline.py`
- Hindi and Marathi text-to-speech output
- Lexical and LLM-based faithfulness checks
- Streamlit interface in `app.py`

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install `ffmpeg` separately if it is not already available.

Place an audio file named `podcast.mp3` in the project root, or update `AUDIO_PATH` in the notebook.

## Run the notebook

Open `Podcast_to_Everything_Pipeline.ipynb` in VS Code with the Jupyter extension and run all cells. Leave the OpenAI key empty to use the local model.

## Run the Streamlit app

```bash
streamlit run app.py
```

Upload an audio file in the browser, select languages, and run the pipeline.

## Deploy on Streamlit Community Cloud

1. Open [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
2. Create an app from `codedbyadiiii/podcast-to-everything`, branch `main`, file `app.py`.
3. In **Advanced settings**, add this secret:

```toml
OPENAI_API_KEY = "your-openai-key"
```

4. Deploy the app and share the generated `streamlit.app` URL.

The deployed app uses OpenAI when `OPENAI_API_KEY` is configured. Do not commit the key to GitHub. The local-model fallback is intended for local use and can exceed the memory limits of free cloud hosting.

## Output

Generated files are written to `outputs/`, including the transcript, summary, key points, social posts, translated text, translated MP3 files, and a faithfulness report. These files are ignored by Git because they can be large and may contain private podcast content.

## Security

Never commit API keys. Set an OpenAI key only through the environment:

```bash
export OPENAI_API_KEY="your-key"
```
