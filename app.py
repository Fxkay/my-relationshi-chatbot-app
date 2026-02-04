import os
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

import streamlit as st
from dotenv import load_dotenv

# Text splitter (modern package)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# Optional text-to-speech (audio)
try:
    from gtts import gTTS
    from io import BytesIO
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False

# Optional microphone input (browser)
try:
    from streamlit_mic_recorder import speech_to_text
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False


# -------------------------------------------------------------------
# Basic config
# -------------------------------------------------------------------
st.set_page_config(page_title="Chity Happy the Love Chatbot",
                   page_icon="🤖", layout="wide")
st.title("🤖 Chitty RAG Chatbot")

st.write(
    "This is **Chitty**, your RAG chatbot. "
    "It reads a text file, stores it in a vector database (Chroma), "
    "and answers questions using retrieved context + an LLM."
)

load_dotenv()

CHROMA_DIR = "./chroma"
COLLECTION = "uploaded_text"
DEFAULT_TEXT_PATH = Path("RAG_TEXT_time.txt")


# -------------------------------------------------------------------
# Your system prompt (QA)
# -------------------------------------------------------------------
QA_SYSTEM_PROMPT = (
    "You are a helpful assistant named Chit "
    "Your knowledge comes from a video from YouTube that discusses Relationships. "
    "Using the timecodes and the context provided, answer the user's question as accurately as possible and trace part of the video "
    "Always start by introducing yourself briefly if the session is new. "
    "You are here to discuss the Relationsips (never call it a 'text file'). "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, say that you don't know. "
    "If you were asked how many questions you have been asked count the number questions and asnwer. "
    "Only use information from the text provided.DO NOT MAKE UP ANSWERS."
    "Give accurate timecodes concerning the knowleadeanswers "
    "You where trained by Ai Engineering Genius Lukman"
    "Lukman was trained by Bashaar from Iron Hack"
    "if asked who is Lukman is , say that he is the AI engineering who trained me"
    "Bashaar trained Lukman on AI engineering at Iron Hack"
    "BE AS CONCISE AS POSSIBLE. "
    "The knowlegde originated from this link = https://www.youtube.com/watch?v=WLKZz6vA0QQ "
    "IMPORTANT: Answer in the same language that the user asks the question. "
    "\n\n"
    "{context}"
)


# -------------------------------------------------------------------
# LLM wrapper classes
# -------------------------------------------------------------------
class OpenAIGenerator:
    """Wrapper around ChatOpenAI that always returns plain text."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)

    def __call__(self, prompt_text: str) -> str:
        msg = self.llm.invoke(prompt_text)
        return getattr(msg, "content", str(msg))


class HFText2TextLLM:
    """Local Flan-T5 generator used when OpenAI is not available."""

    def __init__(self, model_id: str = "google/flan-t5-base"):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        self._pipe = pipeline(
            "text2text-generation",
            model=model,
            tokenizer=tokenizer,
        )

    def __call__(self, prompt_text: str) -> str:
        out = self._pipe(prompt_text, max_new_tokens=256, truncation=True)
        return out[0]["generated_text"]


# -------------------------------------------------------------------
# Cached resources
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_generator(use_openai: bool, temperature: float) -> Tuple[object, str]:
    """
    Build either an OpenAI-based generator (if key present)
    or a local Flan-T5 generator as fallback.
    """
    if use_openai:
        try:
            gen = OpenAIGenerator(model_name="gpt-4o-mini",
                                  temperature=temperature)
            return gen, f"OpenAI gpt-4o-mini (T={temperature:.2f})"
        except Exception as e:
            st.sidebar.error(f"OpenAI initialisation failed: {e}")
            st.sidebar.warning("Falling back to local Flan-T5 model.")

    gen = HFText2TextLLM()
    return gen, "Local google/flan-t5-base (T slider not applied)"


def build_vectorstore(raw_text: str, source_name: str) -> Chroma:
    """
    Chunk text, embed, and store in a persistent Chroma vector store.
    Always recreates the index so it's in sync with the latest text.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(raw_text)
    docs: List[Document] = [
        Document(page_content=c, metadata={"source": source_name})
        for c in chunks
    ]

    chroma_path = Path(CHROMA_DIR)
    if chroma_path.exists():
        shutil.rmtree(chroma_path)

    embeddings = get_embeddings()
    vs = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    return vs


@st.cache_resource(show_spinner=False)
def load_existing_vectorstore() -> Optional[Chroma]:
    """
    Load an existing Chroma brain from disk if present.
    Returns None if nothing is there yet.
    """
    chroma_path = Path(CHROMA_DIR)
    if not chroma_path.exists():
        return None

    embeddings = get_embeddings()
    vs = Chroma(
        embedding_function=embeddings,
        collection_name=COLLECTION,
        persist_directory=CHROMA_DIR,
    )
    return vs


# -------------------------------------------------------------------
# RAG helpers
# -------------------------------------------------------------------
def generate_multiquery_variants(question: str, generator) -> List[str]:
    """
    Ask the LLM to rewrite the question into a few alternative phrasings.
    """
    prompt = f"""
You are a query rewriter for a retrieval system.

Rewrite the user's question into 3 alternative versions that mean the same thing
but use different wording. Return ONLY the 3 rephrasings, each on its own line.

Original question:
{question}
"""
    text = generator(prompt)
    lines = [ln.strip("-• ").strip() for ln in text.splitlines() if ln.strip()]
    rewrites = lines[:3]
    return [question] + rewrites


def retrieve_context(
    vs: Chroma,
    question: str,
    generator,
    use_multiquery: bool,
    k: int,
) -> List[Document]:
    """
    Retrieve relevant chunks from Chroma, optionally using multi-query expansion.
    """
    if not use_multiquery:
        return vs.similarity_search(question, k=k)

    queries = generate_multiquery_variants(question, generator)
    seen = set()
    all_docs: List[Document] = []

    for q in queries:
        docs = vs.similarity_search(q, k=k)
        for d in docs:
            key = (d.page_content, d.metadata.get("source"))
            if key not in seen:
                seen.add(key)
                all_docs.append(d)

    return all_docs[: max(k, min(len(all_docs), 2 * k))]


def build_prompt(
    question: str,
    docs: List[Document],
    history: List[dict],
    enable_memory: bool,
    language_mode: str,
    strict_mode: bool,
) -> Tuple[str, str]:
    """
    Build the full prompt:
    - Uses your QA_SYSTEM_PROMPT with {context}
    - Optional conversation history
    - A line with total number of user questions so far
    - Optional relaxed-mode extra instruction when strict_mode is False
    Returns (prompt_text, tts_lang_code).
    """

    # Number of user questions so far
    num_questions = sum(1 for m in history if m["role"] == "user")

    # Conversation history (if memory enabled)
    if enable_memory and history:
        lines = []
        for msg in history[-8:]:
            role = "User" if msg["role"] == "user" else "Chitty"
            lines.append(f"{role}: {msg['content']}")
        history_text = "\n".join(lines)
    else:
        history_text = "No previous conversation (memory is disabled)."

    # Context chunks
    if docs:
        context_blocks = []
        for i, d in enumerate(docs, start=1):
            src = d.metadata.get("source", "N/A")
            context_blocks.append(f"[{i}] (source: {src})\n{d.page_content}")
        context_text = "\n\n".join(context_blocks)
    else:
        context_text = "No relevant context retrieved from the Relationships text."

    # Start from your QA system prompt
    system_section = QA_SYSTEM_PROMPT.format(context=context_text)

    # Strictness handling
    if strict_mode:
        strict_extra = ""
    else:
        strict_extra = (
            "\n\nRelaxed mode is ON: if the retrieved context does not fully answer the "
            "question, you may add carefully reasoned information from your own knowledge. "
            "When you do this, clearly say that you are going beyond the Relationships text."
        )

    # Language instruction override (if user forces a language)
    if language_mode == "Auto (match user)":
        tts_lang_code = "en"
        language_extra = ""
    else:
        lang_map = {
            "English": "en",
            "French": "fr",
            "Spanish": "es",
            "German": "de",
        }
        tts_lang_code = lang_map.get(language_mode, "en")
        language_extra = (
            f"\n\nOverride the earlier language instruction and always answer in {language_mode}."
        )

    prompt = f"""
{system_section}{strict_extra}{language_extra}

So far, the user has asked {num_questions} questions in this conversation.

Conversation history:
{history_text}

User question:
{question}

Answer:
""".strip()

    return prompt, tts_lang_code


def maybe_speak_answer(answer: str, tts_lang_code: str, autoplay: bool):
    """
    If audio is enabled and gTTS is available, synthesize the answer as audio.
    """
    if not TTS_AVAILABLE:
        st.info(
            "Audio is enabled, but `gTTS` is not installed. "
            "Run `pip install gTTS` to enable voice."
        )
        return

    try:
        tts = gTTS(text=answer, lang=tts_lang_code)
        audio_buffer = BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        st.audio(audio_buffer, format="audio/mp3", autoplay=autoplay)
    except Exception as e:
        st.warning(f"Failed to generate audio: {e}")


# -------------------------------------------------------------------
# SIDEBAR: brain controls + all settings
# -------------------------------------------------------------------
st.sidebar.header("🧠 Knowledge Base")

# Brain mode: existing vs build
brain_mode = st.sidebar.radio(
    "Brain mode",
    options=["Load existing brain (from disk)", "Build / rebuild from file"],
    index=0,
    help="Existing brain uses the saved Chroma index in ./chroma. "
         "Build/rebuild creates a new brain from a text file.",
)

uploaded_file = None
use_default = False

if "vectorstore" not in st.session_state:
    st.session_state["vectorstore"] = None
if "source_name" not in st.session_state:
    st.session_state["source_name"] = None

if brain_mode == "Build / rebuild from file":
    uploaded_file = st.sidebar.file_uploader(
        "Upload UTF-8 .txt file",
        type=["txt"],
        help="Upload a text file to build a new brain.",
        key="kb_upload",
    )
    use_default = st.sidebar.checkbox(
        "Use local RAG_TEXT_time.txt",
        value=True,
        help="If checked and the file exists, it will be used when no upload is provided.",
        key="kb_use_default",
    )

    if st.sidebar.button("📚 Build / Rebuild knowledge base"):
        raw_text = None
        source_name = ""

        if uploaded_file is not None:
            raw_text = uploaded_file.read().decode("utf-8", errors="ignore")
            source_name = uploaded_file.name
        elif use_default and DEFAULT_TEXT_PATH.exists():
            raw_text = DEFAULT_TEXT_PATH.read_text(
                encoding="utf-8", errors="ignore")
            source_name = DEFAULT_TEXT_PATH.name
        else:
            st.sidebar.error(
                "Please upload a `.txt` file or ensure RAG_TEXT_time.txt exists in this folder."
            )

        if raw_text:
            with st.spinner("Building new brain: splitting text, embeddings, Chroma index..."):
                vs = build_vectorstore(raw_text, source_name)
            st.session_state["vectorstore"] = vs
            st.session_state["source_name"] = source_name
            st.session_state["messages"] = []
            st.sidebar.success(f"New brain built from **{source_name}** ✅")

else:  # Load existing brain
    if st.sidebar.button("🧠 Load existing brain"):
        vs = load_existing_vectorstore()
        if vs is None:
            st.sidebar.error(
                "No existing brain found in ./chroma. Please build one first.")
        else:
            st.session_state["vectorstore"] = vs
            st.session_state["source_name"] = f"Existing Chroma index ({COLLECTION})"
            st.sidebar.success("Existing brain loaded from disk ✅")

# -------------------------------------------------------------------
# Chat settings (with audio + strictness + mic toggles)
# -------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Chat Settings")

# OpenAI + temperature
api_key_input = st.sidebar.text_input(
    "OpenAI API Key (optional, for GPT-4o-mini)",
    type="password",
)
if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input.strip()

temperature = st.sidebar.slider(
    "Temperature (creativity)",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="Higher = more creative, lower = more focused/precise (OpenAI only).",
)

use_openai_pref = st.sidebar.checkbox(
    "Use OpenAI (if key is available)",
    value=True,
    help="If off or if no key is provided, the app uses a local Flan-T5 model.",
)

# Language control
language_mode = st.sidebar.selectbox(
    "Answer language",
    options=["Auto (match user)", "English", "French", "Spanish", "German"],
    index=0,
    help="Auto = answer in the same language as the user's question.",
)

# Strictness ON/OFF
strict_mode = st.sidebar.checkbox(
    "Strict mode (only use retrieved context)",
    value=True,
    help="ON = only use the Relationships text. OFF = may add extra knowledge, "
         "but must say when it goes beyond the text.",
)

# Conversational memory
enable_memory = st.sidebar.checkbox(
    "Enable conversational memory",
    value=True,
    help="If off, Chitty ignores previous questions when answering.",
)

# Multi-query feature
use_multiquery = st.sidebar.checkbox(
    "Use Multi-Query retrieval",
    value=False,
    help="On = generate extra paraphrased queries for better recall (slower).",
)

# Audio / talking (audio on/off button)
allow_talking = st.sidebar.checkbox(
    "Audio (text-to-speech, auto-play)",
    value=False,
    help="If on, Chitty will speak the answer automatically.",
)

# Microphone on/off
mic_enabled = st.sidebar.checkbox(
    "Microphone input (talk to Chitty)",
    value=False,
    help="If on, you can speak your question with the mic.",
)

if mic_enabled and not MIC_AVAILABLE:
    st.sidebar.info(
        "Microphone is enabled, but `streamlit-mic-recorder` is not installed.\n"
        "Run `pip install streamlit-mic-recorder` and restart the app."
    )

# Number of chunks
k_results = st.sidebar.slider(
    "Number of chunks to retrieve (k)",
    min_value=1,
    max_value=8,
    value=3,
)

# Show context
show_sources = st.sidebar.checkbox(
    "Show retrieved context chunks",
    value=True,
)

# Clear chat
if st.sidebar.button("🧹 Clear chat"):
    st.session_state["messages"] = []
    st.experimental_rerun()


# -------------------------------------------------------------------
# MAIN PAGE: status + chat
# -------------------------------------------------------------------
st.markdown("### 🧠 Brain status")

vs = st.session_state.get("vectorstore", None)
source_name = st.session_state.get("source_name", None)

if vs is None:
    st.info(
        "No knowledge base loaded. Use the **left sidebar** to load or build a brain.")
else:
    st.success(f"Knowledge base loaded from **{source_name}**.")

# Show current language + audio + strictness + mic on main page
st.markdown(
    f"**Answer language:** `{language_mode}` &nbsp;&nbsp; • "
    f"**Audio:** `{'On' if allow_talking else 'Off'}` &nbsp;&nbsp; • "
    f"**Strict mode:** `{'On' if strict_mode else 'Off'}` &nbsp;&nbsp; • "
    f"**Mic:** `{'On' if mic_enabled else 'Off'}`"
)

st.markdown("### 💬 Chat with Chitty")

if vs is None:
    st.info("👈 Load or build a brain on the left, then start chatting.")
else:
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # show previous messages
    for msg in st.session_state["messages"]:
        with st.chat_message("assistant" if msg["role"] == "assistant" else "user"):
            st.markdown(msg["content"])

    # ----- Input: mic OR text box -----
    user_question: Optional[str] = None

    # 1) Mic input (if enabled and component installed)
    if mic_enabled and MIC_AVAILABLE:
        st.markdown("#### 🎙️ Speak your question")
        voice_text = speech_to_text(
            language="en",  # browser STT language hint
            start_prompt="🎙️ Start talking",
            stop_prompt="🛑 Stop",
            key="chitty_stt",
        )
        if voice_text:
            user_question = voice_text

    # 2) Text input (fallback or additional)
    if user_question is None:
        user_question = st.chat_input(
            "Ask Chitty anything about the Relationships text...")

    # -----------------------------------
    if user_question:
        # store user message
        st.session_state["messages"].append(
            {"role": "user", "content": user_question})
        with st.chat_message("user"):
            st.markdown(user_question)

        # choose LLM
        use_openai_flag = use_openai_pref and bool(
            os.environ.get("OPENAI_API_KEY", "").strip())
        generator, _gen_label = build_generator(use_openai_flag, temperature)

        with st.chat_message("assistant"):
            with st.spinner("🤖 Chitty thinking!!"):
                docs = retrieve_context(
                    vs=vs,
                    question=user_question,
                    generator=generator,
                    use_multiquery=use_multiquery,
                    k=k_results,
                )
                prompt, tts_lang_code = build_prompt(
                    question=user_question,
                    docs=docs,
                    history=st.session_state["messages"],
                    enable_memory=enable_memory,
                    language_mode=language_mode,
                    strict_mode=strict_mode,
                )
                answer = generator(prompt)

            st.markdown(answer)
            st.session_state["messages"].append(
                {"role": "assistant", "content": answer})

            # Audio output if allowed (auto-play)
            if allow_talking:
                maybe_speak_answer(answer, tts_lang_code, autoplay=True)

            # Show retrieved chunks
            if show_sources and docs:
                with st.expander("🔍 Show retrieved context chunks"):
                    for i, d in enumerate(docs, start=1):
                        st.markdown(
                            f"**Chunk {i} (source: {d.metadata.get('source', 'N/A')})**")
                        st.write(d.page_content)
                        st.markdown("---")
