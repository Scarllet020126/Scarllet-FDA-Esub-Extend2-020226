import os
import json
import base64
import hashlib
from datetime import datetime, date
from io import BytesIO
from typing import Any, Dict, List, Optional

import streamlit as st
import yaml
import pandas as pd
import altair as alt
from pypdf import PdfReader

try:
    from docx import Document  # python-docx
except Exception:
    Document = None

try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
except Exception:
    canvas = None
    letter = None

from openai import OpenAI
import google.generativeai as genai
from anthropic import Anthropic
import httpx


# =========================================================
# Models (expanded) + provider routing (pattern-based)
# =========================================================

ALL_MODELS = [
    # OpenAI
    "gpt-4o-mini",
    "gpt-4.1-mini",
    # Gemini
    "gemini-2.5-flash",
    "gemini-3-flash-preview",
    "gemini-2.5-flash-lite",
    "gemini-3-pro-preview",
    # Anthropic (examples; adjust to your available IDs)
    "claude-3-5-sonnet-2024-10",
    "claude-3-5-haiku-20241022",
    # Grok (xAI)
    "grok-4-fast-reasoning",
    "grok-3-mini",
]

# =========================================================
# Painter Styles (20) + localization
# =========================================================

PAINTER_STYLES = [
    "Van Gogh", "Monet", "Picasso", "Da Vinci", "Rembrandt",
    "Matisse", "Kandinsky", "Hokusai", "Yayoi Kusama", "Frida Kahlo",
    "Salvador Dali", "Rothko", "Pollock", "Chagall", "Basquiat",
    "Haring", "Georgia O'Keeffe", "Turner", "Seurat", "Escher",
]

STYLE_CSS = {
    "Van Gogh": "body { background: radial-gradient(circle at top left, #243B55, #141E30); }",
    "Monet": "body { background: linear-gradient(120deg, #a1c4fd, #c2e9fb); }",
    "Picasso": "body { background: linear-gradient(135deg, #ff9a9e, #fecfef); }",
    "Da Vinci": "body { background: radial-gradient(circle, #f9f1c6, #c9a66b); }",
    "Rembrandt": "body { background: radial-gradient(circle, #2c1810, #0b090a); }",
    "Matisse": "body { background: linear-gradient(135deg, #ffecd2, #fcb69f); }",
    "Kandinsky": "body { background: linear-gradient(135deg, #00c6ff, #0072ff); }",
    "Hokusai": "body { background: linear-gradient(135deg, #2b5876, #4e4376); }",
    "Yayoi Kusama": "body { background: radial-gradient(circle, #ffdd00, #ff6a00); }",
    "Frida Kahlo": "body { background: linear-gradient(135deg, #f8b195, #f67280, #c06c84); }",
    "Salvador Dali": "body { background: linear-gradient(135deg, #1a2a6c, #b21f1f, #fdbb2d); }",
    "Rothko": "body { background: linear-gradient(135deg, #141E30, #243B55); }",
    "Pollock": "body { background: repeating-linear-gradient(45deg,#222,#222 10px,#333 10px,#333 20px); }",
    "Chagall": "body { background: linear-gradient(135deg, #a18cd1, #fbc2eb); }",
    "Basquiat": "body { background: linear-gradient(135deg, #f7971e, #ffd200); }",
    "Haring": "body { background: linear-gradient(135deg, #ff512f, #dd2476); }",
    "Georgia O'Keeffe": "body { background: linear-gradient(135deg, #ffefba, #ffffff); }",
    "Turner": "body { background: linear-gradient(135deg, #f8ffae, #43c6ac); }",
    "Seurat": "body { background: radial-gradient(circle, #e0eafc, #cfdef3); }",
    "Escher": "body { background: linear-gradient(135deg, #232526, #414345); }",
}

LABELS = {
    "Dashboard": {"English": "Dashboard", "繁體中文": "儀表板"},
    "TW Premarket": {"English": "TW Premarket Application", "繁體中文": "第二、三等級醫療器材查驗登記"},
    "TFDA Extension": {"English": "TFDA License Extension", "繁體中文": "TFDA 許可證展延（附件/文件）"},
    "510k_tab": {"English": "510(k) Intelligence", "繁體中文": "510(k) 智能分析"},
    "PDF → Markdown": {"English": "PDF → Markdown", "繁體中文": "PDF → Markdown"},
    "Checklist & Report": {"English": "510(k) Review Pipeline", "繁體中文": "510(k) 審查全流程"},
    "Note Keeper & Magics": {"English": "Note Keeper & Magics", "繁體中文": "筆記助手與魔法"},
    "Agents Config": {"English": "Agents Config Studio", "繁體中文": "代理設定工作室"},
    "Clipboard": {"English": "Clipboard", "繁體中文": "剪貼簿"},
}


def t(key: str) -> str:
    lang = st.session_state.settings.get("language", "English")
    return LABELS.get(key, {}).get(lang, key)


def apply_style(theme: str, painter_style: str):
    css = STYLE_CSS.get(painter_style, "")
    if theme == "Dark":
        css += """
        body { color: #e5e7eb; }
        .stButton>button { background-color: #1f2933; color: white; border-radius: 999px; }
        .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div, .stDateInput>div>div>input {
            background-color: #111827; color: #e5e7eb; border-radius: 0.5rem;
        }
        """
    else:
        css += """
        body { color: #111827; }
        .stButton>button { background-color: #2563eb; color: white; border-radius: 999px; }
        .stTextInput>div>div>input, .stTextArea textarea, .stSelectbox>div>div, .stDateInput>div>div>input {
            background-color: #ffffff; color: #111827; border-radius: 0.5rem;
        }
        """

    css += """
    .wow-card {
        border-radius: 18px;
        padding: 14px 18px;
        margin-bottom: 0.75rem;
        box-shadow: 0 14px 35px rgba(15,23,42,0.45);
        color: #f9fafb;
    }
    .wow-card-title {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        opacity: 0.85;
    }
    .wow-card-main {
        font-size: 1.4rem;
        font-weight: 700;
        margin-top: 4px;
    }
    .wow-badge {
        display:inline-flex;
        align-items:center;
        padding:2px 10px;
        border-radius:999px;
        font-size:0.75rem;
        font-weight:600;
        background:rgba(15,23,42,0.2);
        border:1px solid rgba(148,163,184,0.6);
    }
    .subtle {
        opacity: 0.85;
        font-size: 0.92rem;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


# =========================================================
# Provider routing (robust pattern-based)
# =========================================================

def get_provider(model: str) -> str:
    m = (model or "").lower().strip()
    if m.startswith("gpt-"):
        return "openai"
    if m.startswith("gemini-"):
        return "gemini"
    if m.startswith("claude-"):
        return "anthropic"
    if m.startswith("grok-"):
        return "grok"
    # fallback: try to infer by known keywords
    if "openai" in m:
        return "openai"
    raise ValueError(f"Unknown model/provider for model id: {model}")


def call_llm(
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 12000,
    temperature: float = 0.2,
    api_keys: dict | None = None,
) -> str:
    provider = get_provider(model)
    api_keys = api_keys or {}

    def get_key(name: str, env_var: str) -> str:
        return api_keys.get(name) or os.getenv(env_var) or ""

    if provider == "openai":
        key = get_key("openai", "OPENAI_API_KEY")
        if not key:
            raise RuntimeError("Missing OpenAI API key.")
        client = OpenAI(api_key=key)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt or ""},
                {"role": "user", "content": user_prompt or ""},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content

    if provider == "gemini":
        key = get_key("gemini", "GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Missing Gemini API key.")
        genai.configure(api_key=key)
        llm = genai.GenerativeModel(model)
        resp = llm.generate_content(
            (system_prompt or "") + "\n\n" + (user_prompt or ""),
            generation_config={
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            },
        )
        return resp.text

    if provider == "anthropic":
        key = get_key("anthropic", "ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("Missing Anthropic API key.")
        client = Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            system=system_prompt or "",
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": user_prompt or ""}],
        )
        return resp.content[0].text

    if provider == "grok":
        key = get_key("grok", "GROK_API_KEY")
        if not key:
            raise RuntimeError("Missing Grok (xAI) API key.")
        with httpx.Client(base_url="https://api.x.ai/v1", timeout=90) as client:
            resp = client.post(
                "/chat/completions",
                headers={"Authorization": f"Bearer {key}"},
                json={
                    "model": model,
                    "messages": [
                        {"role": "system", "content": system_prompt or ""},
                        {"role": "user", "content": user_prompt or ""},
                    ],
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                },
            )
            resp.raise_for_status()
            data = resp.json()
        return data["choices"][0]["message"]["content"]

    raise RuntimeError(f"Unsupported provider for model {model}")


# =========================================================
# Generic helpers
# =========================================================

def show_status(step_name: str, status: str):
    color = {
        "pending": "gray",
        "running": "#f59e0b",
        "done": "#10b981",
        "error": "#ef4444",
    }.get(status, "gray")
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;margin-bottom:0.25rem;">
          <div style="width:10px;height:10px;border-radius:50%;background:{color};
                      margin-right:6px;"></div>
          <span style="font-size:0.9rem;">{step_name} – <b>{status}</b></span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def log_event(tab: str, agent: str, model: str, tokens_est: int, status: str = "done"):
    st.session_state["history"].append(
        {
            "tab": tab,
            "agent": agent,
            "model": model,
            "tokens_est": int(tokens_est),
            "status": status,
            "ts": datetime.utcnow().isoformat(),
        }
    )


def extract_pdf_pages_to_text(file, start_page: int, end_page: int) -> str:
    reader = PdfReader(file)
    n = len(reader.pages)
    start = max(0, start_page - 1)
    end = min(n, end_page)
    texts = []
    for i in range(start, end):
        try:
            texts.append(reader.pages[i].extract_text() or "")
        except Exception:
            texts.append("")
    return "\n\n".join(texts)


def extract_docx_to_text(file) -> str:
    if Document is None:
        return ""
    try:
        bio = BytesIO(file.read())
        doc = Document(bio)
        return "\n".join(p.text for p in doc.paragraphs)
    except Exception:
        return ""


def create_pdf_from_text(text: str) -> bytes:
    if canvas is None or letter is None:
        raise RuntimeError("reportlab is not installed. Add 'reportlab' to requirements.txt.")
    buf = BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    width, height = letter
    margin = 72
    line_height = 14
    y = height - margin
    for line in (text or "").splitlines():
        if y < margin:
            c.showPage()
            y = height - margin
        c.drawString(margin, y, line[:2000])
        y -= line_height
    c.save()
    buf.seek(0)
    return buf.getvalue()


def show_pdf(pdf_bytes: bytes, height: int = 600):
    if not pdf_bytes:
        return
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    pdf_html = f"""
    <iframe src="data:application/pdf;base64,{b64}"
            width="100%" height="{height}" type="application/pdf"></iframe>
    """
    st.markdown(pdf_html, unsafe_allow_html=True)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


# =========================================================
# Clipboard (agent chaining)
# =========================================================

def ensure_clipboard():
    if "clipboard" not in st.session_state:
        st.session_state["clipboard"] = {"text": "", "source": "", "ts": ""}


def set_clipboard(text: str, source: str):
    ensure_clipboard()
    st.session_state["clipboard"] = {
        "text": text or "",
        "source": source,
        "ts": datetime.utcnow().isoformat(),
    }


def render_clipboard_panel():
    ensure_clipboard()
    cb = st.session_state["clipboard"]
    st.markdown(f"### {t('Clipboard')}")
    st.caption("可將任一 Agent 的輸出送到剪貼簿，再貼到下一個 Agent 的輸入。")
    st.text_area("Clipboard content", value=cb.get("text", ""), height=140, key="clipboard_view")
    st.caption(f"Source: {cb.get('source','')} · Time(UTC): {cb.get('ts','')}")


# =========================================================
# Agent UI runner (editable prompt/model/tokens + clipboard)
# =========================================================

def agent_run_ui(
    agent_id: str,
    tab_key: str,
    default_prompt: str,
    default_input_text: str = "",
    allow_model_override: bool = True,
    tab_label_for_history: str | None = None,
):
    agents_cfg = st.session_state.get("agents_cfg", {})
    agents_dict = agents_cfg.get("agents", {})

    agent_cfg = agents_dict.get(agent_id, {
        "name": agent_id,
        "model": st.session_state.settings["model"],
        "system_prompt": "",
        "max_tokens": st.session_state.settings["max_tokens"],
        "temperature": st.session_state.settings["temperature"],
    })

    status_key = f"{tab_key}_status"
    if status_key not in st.session_state:
        st.session_state[status_key] = "pending"

    show_status(agent_cfg.get("name", agent_id), st.session_state[status_key])

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        user_prompt = st.text_area(
            "Prompt",
            value=st.session_state.get(f"{tab_key}_prompt", default_prompt),
            height=160,
            key=f"{tab_key}_prompt",
        )
    with col2:
        base_model = st.session_state.get(f"{tab_key}_model", agent_cfg.get("model", st.session_state.settings["model"]))
        if base_model not in ALL_MODELS:
            # allow typing unknown model ids via fallback selectbox behavior:
            base_model = st.session_state.settings["model"]
        model = st.selectbox(
            "Model",
            ALL_MODELS,
            index=ALL_MODELS.index(base_model),
            disabled=not allow_model_override,
            key=f"{tab_key}_model",
        )
    with col3:
        max_tokens = st.number_input(
            "max_tokens",
            min_value=1000,
            max_value=120000,
            value=int(st.session_state.get(f"{tab_key}_max_tokens", agent_cfg.get("max_tokens", st.session_state.settings["max_tokens"]))),
            step=1000,
            key=f"{tab_key}_max_tokens",
        )

    # clipboard helpers
    ensure_clipboard()
    cb_col1, cb_col2 = st.columns([1, 1])
    with cb_col1:
        if st.button("Load from Clipboard → Input", key=f"{tab_key}_load_cb"):
            st.session_state[f"{tab_key}_input"] = st.session_state["clipboard"]["text"] or default_input_text
    with cb_col2:
        if st.button("Clear Input", key=f"{tab_key}_clear_input"):
            st.session_state[f"{tab_key}_input"] = ""

    input_text = st.text_area(
        "Input Text / Markdown",
        value=st.session_state.get(f"{tab_key}_input", default_input_text),
        height=260,
        key=f"{tab_key}_input",
    )

    run = st.button("Run Agent", key=f"{tab_key}_run")

    if run:
        st.session_state[status_key] = "running"
        show_status(agent_cfg.get("name", agent_id), "running")
        api_keys = st.session_state.get("api_keys", {})
        system_prompt = agent_cfg.get("system_prompt", "")

        user_full = f"{user_prompt}\n\n---\n\n{input_text}"

        with st.spinner("Running agent..."):
            try:
                out = call_llm(
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_full,
                    max_tokens=int(max_tokens),
                    temperature=float(st.session_state.settings["temperature"]),
                    api_keys=api_keys,
                )
                st.session_state[f"{tab_key}_output"] = out
                st.session_state[status_key] = "done"
                token_est = int(len((user_full or "") + (out or "")) / 4)
                log_event(
                    tab_label_for_history or tab_key,
                    agent_cfg.get("name", agent_id),
                    model,
                    token_est,
                    status="done",
                )
            except Exception as e:
                st.session_state[status_key] = "error"
                log_event(tab_label_for_history or tab_key, agent_cfg.get("name", agent_id), model, 0, status="error")
                st.error(f"Agent error: {e}")

    output = st.session_state.get(f"{tab_key}_output", "")
    view_mode = st.radio(
        "View mode",
        ["Markdown", "Plain text"],
        horizontal=True,
        key=f"{tab_key}_viewmode",
    )

    edited = st.text_area(
        "Output (editable)",
        value=output,
        height=320,
        key=f"{tab_key}_output_edited",
    )

    col_out1, col_out2 = st.columns([1, 1])
    with col_out1:
        if st.button("Send edited output → Clipboard", key=f"{tab_key}_to_cb"):
            set_clipboard(edited, source=f"{tab_label_for_history or tab_key} · {agent_cfg.get('name', agent_id)}")
            st.success("已送到剪貼簿。")
    with col_out2:
        if view_mode == "Markdown" and edited.strip():
            st.markdown("Preview")
            st.markdown(edited, unsafe_allow_html=True)


# =========================================================
# Sidebar
# =========================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### Global Settings")

        st.session_state.settings["theme"] = st.radio(
            "Theme", ["Light", "Dark"],
            index=0 if st.session_state.settings["theme"] == "Light" else 1,
        )

        st.session_state.settings["language"] = st.radio(
            "Language", ["English", "繁體中文"],
            index=0 if st.session_state.settings["language"] == "English" else 1,
        )

        col1, col2 = st.columns([3, 1])
        with col1:
            style = st.selectbox(
                "Painter Style",
                PAINTER_STYLES,
                index=PAINTER_STYLES.index(st.session_state.settings["painter_style"]),
            )
        with col2:
            if st.button("Jackpot!"):
                import random
                style = random.choice(PAINTER_STYLES)
        st.session_state.settings["painter_style"] = style

        st.session_state.settings["model"] = st.selectbox(
            "Default Model",
            ALL_MODELS,
            index=ALL_MODELS.index(st.session_state.settings["model"]),
        )
        st.session_state.settings["max_tokens"] = st.number_input(
            "Default max_tokens",
            min_value=1000,
            max_value=120000,
            value=int(st.session_state.settings["max_tokens"]),
            step=1000,
        )
        st.session_state.settings["temperature"] = st.slider(
            "Temperature",
            0.0,
            1.0,
            float(st.session_state.settings["temperature"]),
            0.05,
        )

        st.markdown("---")
        st.markdown("### API Keys")

        keys: Dict[str, str] = {}

        # IMPORTANT: do not show env key; just indicate presence.
        if os.getenv("OPENAI_API_KEY"):
            st.caption("OpenAI key loaded from environment.")
        else:
            keys["openai"] = st.text_input("OpenAI API Key", type="password")

        if os.getenv("GEMINI_API_KEY"):
            st.caption("Gemini key loaded from environment.")
        else:
            keys["gemini"] = st.text_input("Gemini API Key", type="password")

        if os.getenv("ANTHROPIC_API_KEY"):
            st.caption("Anthropic key loaded from environment.")
        else:
            keys["anthropic"] = st.text_input("Anthropic API Key", type="password")

        if os.getenv("GROK_API_KEY"):
            st.caption("Grok key loaded from environment.")
        else:
            keys["grok"] = st.text_input("Grok API Key", type="password")

        st.session_state["api_keys"] = keys

        st.markdown("---")
        st.markdown("### Demo Data (Mock Cases)")
        case = st.selectbox(
            "Load a mock TFDA case",
            ["(none)", "Mock Case A", "Mock Case B", "Mock Case C"],
            index=0,
            key="mock_case_select",
        )
        if st.button("Load selected mock case", key="mock_case_load_btn"):
            if case != "(none)":
                load_mock_case(case)
                st.success(f"Loaded: {case}")
                st.rerun()

        st.markdown("---")
        st.markdown("### Agents Catalog (agents.yaml)")
        uploaded_agents = st.file_uploader(
            "Upload custom agents.yaml",
            type=["yaml", "yml"],
            key="sidebar_agents_yaml",
        )
        if uploaded_agents is not None:
            try:
                cfg = yaml.safe_load(uploaded_agents.read())
                if "agents" in cfg:
                    st.session_state["agents_cfg"] = cfg
                    st.success("Custom agents.yaml loaded for this session.")
                else:
                    st.warning("Uploaded YAML has no top-level 'agents' key. Using previous config.")
            except Exception as e:
                st.error(f"Failed to parse uploaded YAML: {e}")

        st.markdown("---")
        render_clipboard_panel()


# =========================================================
# Dashboard (Awesome)
# =========================================================

def render_dashboard():
    st.title(t("Dashboard"))
    hist = st.session_state["history"]
    if not hist:
        st.info("No runs yet.")
        return

    df = pd.DataFrame(hist)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Runs", len(df))
    with col2:
        st.metric("Unique Tabs", df["tab"].nunique())
    with col3:
        st.metric("Approx Tokens", int(df["tokens_est"].sum()))
    with col4:
        err_cnt = int((df["status"] == "error").sum()) if "status" in df.columns else 0
        st.metric("Errors", err_cnt)

    # Provider readiness
    st.markdown("### Operational Health")
    env_openai = bool(os.getenv("OPENAI_API_KEY")) or bool(st.session_state["api_keys"].get("openai"))
    env_gemini = bool(os.getenv("GEMINI_API_KEY")) or bool(st.session_state["api_keys"].get("gemini"))
    env_anth = bool(os.getenv("ANTHROPIC_API_KEY")) or bool(st.session_state["api_keys"].get("anthropic"))
    env_grok = bool(os.getenv("GROK_API_KEY")) or bool(st.session_state["api_keys"].get("grok"))

    health = pd.DataFrame([{
        "Provider": "OpenAI", "Ready": env_openai,
    }, {
        "Provider": "Gemini", "Ready": env_gemini,
    }, {
        "Provider": "Anthropic", "Ready": env_anth,
    }, {
        "Provider": "Grok", "Ready": env_grok,
    }])
    st.dataframe(health, use_container_width=True, hide_index=True)

    st.markdown("### WOW Status Wall – Latest Activity")
    last = df.sort_values("ts", ascending=False).iloc[0]
    wow_color = "linear-gradient(135deg,#22c55e,#16a34a)"
    if last["tokens_est"] > 40000:
        wow_color = "linear-gradient(135deg,#f97316,#ea580c)"
    if last["tokens_est"] > 80000:
        wow_color = "linear-gradient(135deg,#ef4444,#b91c1c)"

    st.markdown(
        f"""
        <div class="wow-card" style="background:{wow_color};">
          <div class="wow-card-title">LATEST RUN SNAPSHOT</div>
          <div class="wow-card-main">
            {last['tab']} · {last['agent']}
          </div>
          <div style="margin-top:6px;font-size:0.9rem;">
            Model: <b>{last['model']}</b> · Tokens ≈ <b>{last['tokens_est']}</b><br>
            Time (UTC): {last['ts']} · Status: <b>{last.get('status','')}</b>
          </div>
          <div style="margin-top:8px;">
            <span class="wow-badge">Theme: {st.session_state.settings['theme']}</span>
            <span class="wow-badge" style="margin-left:6px;">Lang: {st.session_state.settings['language']}</span>
            <span class="wow-badge" style="margin-left:6px;">Style: {st.session_state.settings['painter_style']}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### Runs by Tab")
    chart_tab = alt.Chart(df).mark_bar().encode(
        x="tab:N", y="count():Q", color="tab:N", tooltip=["tab", "count()"],
    )
    st.altair_chart(chart_tab, use_container_width=True)

    st.markdown("### Runs by Model")
    chart_model = alt.Chart(df).mark_bar().encode(
        x="model:N", y="count():Q", color="model:N", tooltip=["model", "count()"],
    )
    st.altair_chart(chart_model, use_container_width=True)

    st.markdown("### Model × Tab Usage Heatmap")
    heat_df = df.groupby(["tab", "model"]).size().reset_index(name="count")
    heatmap = (
        alt.Chart(heat_df)
        .mark_rect()
        .encode(
            x=alt.X("model:N", title="Model"),
            y=alt.Y("tab:N", title="Tab"),
            color=alt.Color("count:Q", scale=alt.Scale(scheme="blues"), title="Runs"),
            tooltip=["tab", "model", "count"],
        )
        .properties(height=260)
    )
    st.altair_chart(heatmap, use_container_width=True)

    st.markdown("### Token Usage Over Time")
    df_time = df.copy()
    df_time["ts"] = pd.to_datetime(df_time["ts"])
    chart_time = alt.Chart(df_time).mark_line(point=True).encode(
        x="ts:T", y="tokens_est:Q", color="tab:N",
        tooltip=["ts", "tab", "agent", "model", "tokens_est", "status"],
    )
    st.altair_chart(chart_time, use_container_width=True)

    st.markdown("### Recent Activity")
    st.dataframe(df.sort_values("ts", ascending=False).head(25), use_container_width=True)


# =========================================================
# Existing TFDA TW premarket (kept; minimal changes)
# =========================================================

TW_APP_FIELDS = [
    "doc_no", "e_no", "apply_date", "case_type", "device_category", "case_kind",
    "origin", "product_class", "similar", "replace_flag", "prior_app_no",
    "name_zh", "name_en", "indications", "spec_comp",
    "main_cat", "item_code", "item_name",
    "uniform_id", "firm_name", "firm_addr",
    "resp_name", "contact_name", "contact_tel", "contact_fax", "contact_email",
    "confirm_match", "cert_raps", "cert_ahwp", "cert_other",
    "manu_type", "manu_name", "manu_country", "manu_addr", "manu_note",
    "auth_applicable", "auth_desc",
    "cfs_applicable", "cfs_desc",
    "qms_applicable", "qms_desc",
    "similar_info", "labeling_info", "tech_file_info",
    "preclinical_info", "preclinical_replace",
    "clinical_just", "clinical_info",
]


def build_tw_app_dict_from_session() -> dict:
    s = st.session_state
    apply_date = s.get("tw_apply_date")
    apply_date_str = apply_date.strftime("%Y-%m-%d") if apply_date else ""
    return {
        "doc_no": s.get("tw_doc_no", ""),
        "e_no": s.get("tw_e_no", ""),
        "apply_date": apply_date_str,
        "case_type": s.get("tw_case_type", ""),
        "device_category": s.get("tw_device_category", ""),
        "case_kind": s.get("tw_case_kind", ""),
        "origin": s.get("tw_origin", ""),
        "product_class": s.get("tw_product_class", ""),
        "similar": s.get("tw_similar", ""),
        "replace_flag": s.get("tw_replace_flag", ""),
        "prior_app_no": s.get("tw_prior_app_no", ""),
        "name_zh": s.get("tw_dev_name_zh", ""),
        "name_en": s.get("tw_dev_name_en", ""),
        "indications": s.get("tw_indications", ""),
        "spec_comp": s.get("tw_spec_comp", ""),
        "main_cat": s.get("tw_main_cat", ""),
        "item_code": s.get("tw_item_code", ""),
        "item_name": s.get("tw_item_name", ""),
        "uniform_id": s.get("tw_uniform_id", ""),
        "firm_name": s.get("tw_firm_name", ""),
        "firm_addr": s.get("tw_firm_addr", ""),
        "resp_name": s.get("tw_resp_name", ""),
        "contact_name": s.get("tw_contact_name", ""),
        "contact_tel": s.get("tw_contact_tel", ""),
        "contact_fax": s.get("tw_contact_fax", ""),
        "contact_email": s.get("tw_contact_email", ""),
        "confirm_match": bool(s.get("tw_confirm_match", False)),
        "cert_raps": bool(s.get("tw_cert_raps", False)),
        "cert_ahwp": bool(s.get("tw_cert_ahwp", False)),
        "cert_other": s.get("tw_cert_other", ""),
        "manu_type": s.get("tw_manu_type", ""),
        "manu_name": s.get("tw_manu_name", ""),
        "manu_country": s.get("tw_manu_country", ""),
        "manu_addr": s.get("tw_manu_addr", ""),
        "manu_note": s.get("tw_manu_note", ""),
        "auth_applicable": s.get("tw_auth_app", ""),
        "auth_desc": s.get("tw_auth_desc", ""),
        "cfs_applicable": s.get("tw_cfs_app", ""),
        "cfs_desc": s.get("tw_cfs_desc", ""),
        "qms_applicable": s.get("tw_qms_app", ""),
        "qms_desc": s.get("tw_qms_desc", ""),
        "similar_info": s.get("tw_similar_info", ""),
        "labeling_info": s.get("tw_labeling_info", ""),
        "tech_file_info": s.get("tw_tech_file_info", ""),
        "preclinical_info": s.get("tw_preclinical_info", ""),
        "preclinical_replace": s.get("tw_preclinical_replace", ""),
        "clinical_just": s.get("tw_clinical_app", ""),
        "clinical_info": s.get("tw_clinical_info", ""),
    }


def apply_tw_app_dict_to_session(data: dict):
    s = st.session_state
    s["tw_doc_no"] = data.get("doc_no", "")
    s["tw_e_no"] = data.get("e_no", "MDE")
    try:
        if data.get("apply_date"):
            y, m, d = map(int, str(data["apply_date"]).split("-"))
            s["tw_apply_date"] = date(y, m, d)
    except Exception:
        pass

    s["tw_case_type"] = data.get("case_type", "")
    s["tw_device_category"] = data.get("device_category", "")
    s["tw_case_kind"] = data.get("case_kind", "")
    s["tw_origin"] = data.get("origin", "")
    s["tw_product_class"] = data.get("product_class", "")
    s["tw_similar"] = data.get("similar", "")
    s["tw_replace_flag"] = data.get("replace_flag", "")
    s["tw_prior_app_no"] = data.get("prior_app_no", "")
    s["tw_dev_name_zh"] = data.get("name_zh", "")
    s["tw_dev_name_en"] = data.get("name_en", "")
    s["tw_indications"] = data.get("indications", "")
    s["tw_spec_comp"] = data.get("spec_comp", "")
    s["tw_main_cat"] = data.get("main_cat", "")
    s["tw_item_code"] = data.get("item_code", "")
    s["tw_item_name"] = data.get("item_name", "")
    s["tw_uniform_id"] = data.get("uniform_id", "")
    s["tw_firm_name"] = data.get("firm_name", "")
    s["tw_firm_addr"] = data.get("firm_addr", "")
    s["tw_resp_name"] = data.get("resp_name", "")
    s["tw_contact_name"] = data.get("contact_name", "")
    s["tw_contact_tel"] = data.get("contact_tel", "")
    s["tw_contact_fax"] = data.get("contact_fax", "")
    s["tw_contact_email"] = data.get("contact_email", "")
    s["tw_confirm_match"] = bool(data.get("confirm_match", False))
    s["tw_cert_raps"] = bool(data.get("cert_raps", False))
    s["tw_cert_ahwp"] = bool(data.get("cert_ahwp", False))
    s["tw_cert_other"] = data.get("cert_other", "")
    s["tw_manu_type"] = data.get("manu_type", "")
    s["tw_manu_name"] = data.get("manu_name", "")
    s["tw_manu_country"] = data.get("manu_country", "")
    s["tw_manu_addr"] = data.get("manu_addr", "")
    s["tw_manu_note"] = data.get("manu_note", "")
    s["tw_auth_app"] = data.get("auth_applicable", "")
    s["tw_auth_desc"] = data.get("auth_desc", "")
    s["tw_cfs_app"] = data.get("cfs_applicable", "")
    s["tw_cfs_desc"] = data.get("cfs_desc", "")
    s["tw_qms_app"] = data.get("qms_applicable", "")
    s["tw_qms_desc"] = data.get("qms_desc", "")
    s["tw_similar_info"] = data.get("similar_info", "")
    s["tw_labeling_info"] = data.get("labeling_info", "")
    s["tw_tech_file_info"] = data.get("tech_file_info", "")
    s["tw_preclinical_info"] = data.get("preclinical_info", "")
    s["tw_preclinical_replace"] = data.get("preclinical_replace", "")
    s["tw_clinical_app"] = data.get("clinical_just", "")
    s["tw_clinical_info"] = data.get("clinical_info", "")


def standardize_tw_app_info_with_llm(raw_obj) -> dict:
    api_keys = st.session_state.get("api_keys", {})
    model = "gemini-2.5-flash"
    if not (api_keys.get("gemini") or os.getenv("GEMINI_API_KEY")):
        raise RuntimeError("No Gemini API key available for standardizing application info.")

    raw_json = json.dumps(raw_obj, ensure_ascii=False, indent=2)
    fields_str = ", ".join(TW_APP_FIELDS)

    system_prompt = f"""
你是一位資料標準化助手，協助將任意 JSON/CSV 欄位映射為臺灣 TFDA 第二、三等級醫療器材查驗登記的標準 JSON。

目標：
輸出必須是「單一 JSON 物件」，且必須包含以下所有 keys（除布林欄位外皆為字串）：
{fields_str}

規則：
- 僅輸出 JSON，不得輸出 Markdown/註解。
- 若找不到對應資料：字串欄位填空字串；布林欄位填 false。
- 不得杜撰新資訊，只能重新命名/重組既有資訊。
- apply_date 若可推斷，請輸出 YYYY-MM-DD，否則空字串。
"""

    user_prompt = f"以下是待標準化資料：\n\n{raw_json}"

    out = call_llm(
        model=model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        max_tokens=4000,
        temperature=0.1,
        api_keys=api_keys,
    )

    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        start = out.find("{")
        end = out.rfind("}")
        if start != -1 and end != -1 and end > start:
            data = json.loads(out[start:end + 1])
        else:
            raise RuntimeError("LLM did not return valid JSON.")
    if not isinstance(data, dict):
        raise RuntimeError("Standardized application info is not a JSON object.")
    for k in TW_APP_FIELDS:
        if k not in data:
            data[k] = False if k in ("confirm_match", "cert_raps", "cert_ahwp") else ""
    return data


def compute_tw_app_completeness() -> float:
    s = st.session_state
    required_keys = [
        "tw_e_no", "tw_case_type", "tw_device_category",
        "tw_origin", "tw_product_class",
        "tw_dev_name_zh", "tw_dev_name_en",
        "tw_uniform_id", "tw_firm_name", "tw_firm_addr",
        "tw_resp_name", "tw_contact_name", "tw_contact_tel",
        "tw_contact_email",
        "tw_manu_name", "tw_manu_addr",
    ]
    filled = 0
    for k in required_keys:
        v = s.get(k, "")
        if isinstance(v, str):
            if v.strip():
                filled += 1
        else:
            if v:
                filled += 1
    return filled / len(required_keys) if required_keys else 0.0


def render_tw_premarket_tab():
    st.title(t("TW Premarket"))
    st.caption("保留原功能：匯入/匯出、線上填寫、產出申請書 Markdown、載入指引、Agent 預審、Agent 編修。")

    st.markdown("### Application Info 匯入 / 匯出 (JSON / CSV)")
    col_ie1, col_ie2 = st.columns(2)

    with col_ie1:
        app_file = st.file_uploader("Upload Application Info (JSON / CSV)", type=["json", "csv"], key="tw_app_upload")
        if app_file is not None:
            try:
                if app_file.name.lower().endswith(".json"):
                    raw_data = json.load(app_file)
                else:
                    df = pd.read_csv(app_file)
                    raw_data = df.to_dict(orient="records")[0] if len(df) else {}
                if isinstance(raw_data, dict) and all(k in raw_data for k in TW_APP_FIELDS):
                    standardized = raw_data
                else:
                    with st.spinner("使用 LLM 將欄位轉為標準 TFDA 申請書格式..."):
                        standardized = standardize_tw_app_info_with_llm(raw_data)
                apply_tw_app_dict_to_session(standardized)
                st.session_state["tw_app_last_loaded"] = standardized
                st.success("已套用至申請表單。")
                st.rerun()
            except Exception as e:
                st.error(f"上傳或標準化失敗：{e}")

    with col_ie2:
        app_dict = build_tw_app_dict_from_session()
        st.download_button(
            "Download JSON",
            data=json.dumps(app_dict, ensure_ascii=False, indent=2).encode("utf-8"),
            file_name="tw_premarket_application.json",
            mime="application/json",
        )
        df_app = pd.DataFrame([app_dict])
        st.download_button(
            "Download CSV",
            data=df_app.to_csv(index=False).encode("utf-8"),
            file_name="tw_premarket_application.csv",
            mime="text/csv",
        )

    if "tw_app_last_loaded" in st.session_state:
        st.markdown("**最近載入/標準化之 Application JSON 預覽**")
        st.json(st.session_state["tw_app_last_loaded"], expanded=False)

    st.markdown("---")

    completeness = compute_tw_app_completeness()
    pct = int(completeness * 100)
    if pct >= 80:
        card_grad = "linear-gradient(135deg,#22c55e,#16a34a)"
        txt = "申請基本欄位完成度高，適合進行預審。"
    elif pct >= 50:
        card_grad = "linear-gradient(135deg,#f97316,#ea580c)"
        txt = "部分關鍵欄位仍待補齊，建議補足後再送預審。"
    else:
        card_grad = "linear-gradient(135deg,#ef4444,#b91c1c)"
        txt = "多數基本欄位尚未填寫，請先充實申請資訊。"

    st.markdown(
        f"""
        <div class="wow-card" style="background:{card_grad};">
          <div class="wow-card-title">APPLICATION COMPLETENESS</div>
          <div class="wow-card-main">{pct}%</div>
          <div class="subtle" style="margin-top:6px;">{txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(completeness)

    st.markdown("### Step 1 – 線上填寫申請書（草稿）")
    if "tw_app_status" not in st.session_state:
        st.session_state["tw_app_status"] = "pending"
    show_status("申請書填寫", st.session_state["tw_app_status"])

    # Basic fields (kept)
    st.markdown("#### 一、案件基本資料")
    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        doc_no = st.text_input("公文文號", key="tw_doc_no")
        e_no = st.text_input("電子流水號", value=st.session_state.get("tw_e_no", "MDE"), key="tw_e_no")
    with col_a2:
        apply_date = st.date_input("申請日", key="tw_apply_date")
        case_type = st.selectbox(
            "案件類型*",
            ["一般申請案", "同一產品不同品名", "專供外銷", "許可證有效期限屆至後六個月內重新申請"],
            key="tw_case_type",
        )
    with col_a3:
        device_category = st.selectbox("醫療器材類型*", ["一般醫材", "體外診斷器材(IVD)"], key="tw_device_category")
        case_kind = st.selectbox("案件種類*", ["新案", "變更案", "展延案"], index=0, key="tw_case_kind")

    col_a4, col_a5, col_a6 = st.columns(3)
    with col_a4:
        origin = st.selectbox("產地*", ["國產", "輸入", "陸輸"], key="tw_origin")
    with col_a5:
        product_class = st.selectbox("產品等級*", ["第二等級", "第三等級"], key="tw_product_class")
    with col_a6:
        similar = st.selectbox("有無類似品*", ["有", "無", "全球首創"], key="tw_similar")

    col_a7, col_a8 = st.columns(2)
    with col_a7:
        replace_flag = st.radio(
            "是否勾選「替代臨床前測試及原廠品質管制資料」？*",
            ["否", "是"],
            index=0 if st.session_state.get("tw_replace_flag", "否") == "否" else 1,
            key="tw_replace_flag",
        )
    with col_a8:
        prior_app_no = st.text_input("（非首次申請）前次申請案號", key="tw_prior_app_no")

    st.markdown("#### 二、醫療器材基本資訊")
    col_b1, col_b2 = st.columns(2)
    with col_b1:
        name_zh = st.text_input("醫療器材中文名稱*", key="tw_dev_name_zh")
        name_en = st.text_input("醫療器材英文名稱*", key="tw_dev_name_en")
    with col_b2:
        indications = st.text_area("效能、用途或適應症說明", value=st.session_state.get("tw_indications", "詳如核定之中文說明書"), key="tw_indications")
        spec_comp = st.text_area("型號、規格或主要成分說明", value=st.session_state.get("tw_spec_comp", "詳如核定之中文說明書"), key="tw_spec_comp")

    st.markdown("**分類分級品項（依附表填列）**")
    col_b3, col_b4, col_b5 = st.columns(3)
    with col_b3:
        main_cat = st.selectbox("主類別", ["", "J.一般醫院及個人使用裝置", "P.放射學科學", "A.臨床化學及臨床毒理學"], key="tw_main_cat")
    with col_b4:
        item_code = st.text_input("分級品項代碼（例：J.1234）", key="tw_item_code")
    with col_b5:
        item_name = st.text_input("分級品項名稱", key="tw_item_name")

    st.markdown("#### 三、醫療器材商資料")
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        uniform_id = st.text_input("統一編號*", key="tw_uniform_id")
        firm_name = st.text_input("醫療器材商名稱*", key="tw_firm_name")
        firm_addr = st.text_area("醫療器材商地址*", height=80, key="tw_firm_addr")
    with col_c2:
        resp_name = st.text_input("負責人姓名*", key="tw_resp_name")
        contact_name = st.text_input("聯絡人姓名*", key="tw_contact_name")
        contact_tel = st.text_input("電話*", key="tw_contact_tel")
        contact_fax = st.text_input("聯絡人傳真", key="tw_contact_fax")
        contact_email = st.text_input("電子郵件*", key="tw_contact_email")

    confirm_match = st.checkbox("我已確認上述資料與最新版醫療器材商證照資訊相符", key="tw_confirm_match")

    st.markdown("#### 四、製造廠資訊（含委託製造）")
    manu_type = st.radio(
        "製造方式",
        ["單一製造廠", "全部製程委託製造", "委託非全部製程之製造/包裝/貼標/滅菌及最終驗放"],
        index=0,
        key="tw_manu_type",
    )
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        manu_name = st.text_input("製造廠名稱*", key="tw_manu_name")
        manu_country = st.selectbox("製造國別*", ["TAIWAN， ROC", "UNITED STATES", "JAPAN", "OTHER"], key="tw_manu_country")
    with col_d2:
        manu_addr = st.text_area("製造廠地址*", height=80, key="tw_manu_addr")
        manu_note = st.text_area("製造廠相關說明", height=80, key="tw_manu_note")

    with st.expander("附件摘要（可後續補充）", expanded=False):
        st.selectbox("原廠授權登記書", ["不適用", "適用"], key="tw_auth_app")
        st.text_area("原廠授權登記書說明", height=80, key="tw_auth_desc")
        st.selectbox("出產國製售證明", ["不適用", "適用"], key="tw_cfs_app")
        st.text_area("出產國製售證明說明", height=80, key="tw_cfs_desc")
        st.selectbox("QMS/QSD", ["不適用", "適用"], key="tw_qms_app")
        st.text_area("QMS/QSD 說明", height=80, key="tw_qms_desc")

        st.text_area("類似品與比較表摘要", height=80, key="tw_similar_info")
        st.text_area("標籤／說明書／包裝擬稿重點", height=100, key="tw_labeling_info")
        st.text_area("技術檔案摘要", height=120, key="tw_tech_file_info")
        st.text_area("臨床前測試摘要", height=140, key="tw_preclinical_info")
        st.text_area("替代臨床前測試說明", height=100, key="tw_preclinical_replace")
        st.selectbox("臨床證據是否適用？", ["不適用", "適用"], key="tw_clinical_app")
        st.text_area("臨床證據摘要", height=140, key="tw_clinical_info")

    # Generate application markdown
    if st.button("生成申請書 Markdown 草稿", key="tw_generate_md_btn"):
        missing = []
        for label, val in [
            ("電子流水號", e_no), ("案件類型", case_type), ("醫療器材類型", device_category),
            ("產地", origin), ("產品等級", product_class), ("中文名稱", name_zh), ("英文名稱", name_en),
            ("統一編號", uniform_id), ("器材商名稱", firm_name), ("器材商地址", firm_addr),
            ("負責人姓名", resp_name), ("聯絡人姓名", contact_name), ("電話", contact_tel),
            ("電子郵件", contact_email), ("製造廠名稱", manu_name), ("製造廠地址", manu_addr),
        ]:
            if not (val or "").strip():
                missing.append(label)

        st.session_state["tw_app_status"] = "done" if not missing else "error"
        if missing:
            st.warning("以下基本欄位尚未填寫完整（形式檢查）：\n- " + "\n- ".join(missing))

        apply_date_str = apply_date.strftime("%Y-%m-%d") if apply_date else ""

        st.session_state["tw_app_markdown"] = f"""# 第二、三等級醫療器材查驗登記申請書（線上草稿）

## 一、案件基本資料
- 公文文號：{doc_no or "（未填）"}
- 電子流水號：{e_no or "（未填）"}
- 申請日：{apply_date_str or "（未填）"}
- 案件類型：{case_type}
- 醫療器材類型：{device_category}
- 案件種類：{case_kind}
- 產地：{origin}
- 產品等級：{product_class}
- 有無類似品：{similar}
- 替代條款勾選：{replace_flag}
- 前次申請案號（如適用）：{prior_app_no or "不適用"}

## 二、醫療器材基本資訊
- 中文名稱：{name_zh}
- 英文名稱：{name_en}
- 效能、用途或適應症：{indications}
- 型號/規格/主要成分：{spec_comp}

### 分類分級品項
- 主類別：{main_cat or "（未填）"}
- 分級品項代碼：{item_code or "（未填）"}
- 分級品項名稱：{item_name or "（未填）"}

## 三、醫療器材商資料
- 統一編號：{uniform_id}
- 名稱：{firm_name}
- 地址：{firm_addr}
- 負責人：{resp_name}
- 聯絡人：{contact_name}
- 電話：{contact_tel}
- 傳真：{contact_fax or "（未填）"}
- Email：{contact_email}
- 與最新證照資訊相符：{"是" if confirm_match else "否"}

## 四、製造廠資訊
- 製造方式：{manu_type}
- 製造廠名稱：{manu_name}
- 製造國別：{manu_country}
- 製造廠地址：{manu_addr}
- 製造相關說明：{manu_note or "（未填）"}
"""

    st.markdown("##### 申請書 Markdown（可編輯）")
    app_md_current = st.session_state.get("tw_app_markdown", "")
    mode = st.radio("檢視模式", ["Markdown", "純文字"], horizontal=True, key="tw_app_viewmode")
    app_md_edited = st.text_area("申請書內容", value=app_md_current, height=280, key="tw_app_md_edited")
    st.session_state["tw_app_effective_md"] = app_md_edited

    st.markdown("---")
    st.markdown("### Step 2 – 輸入預審/形式審查指引（Screen Review Guidance）")
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        guidance_file = st.file_uploader("上傳預審指引 (PDF / TXT / MD)", type=["pdf", "txt", "md"], key="tw_guidance_file")
        guidance_text_from_file = ""
        if guidance_file is not None:
            suffix = guidance_file.name.lower().rsplit(".", 1)[-1]
            if suffix == "pdf":
                guidance_text_from_file = extract_pdf_pages_to_text(guidance_file, 1, 9999)
            else:
                guidance_text_from_file = guidance_file.read().decode("utf-8", errors="ignore")
    with col_g2:
        guidance_text_manual = st.text_area("或直接貼上指引文字/Markdown", height=180, key="tw_guidance_manual")

    guidance_text = guidance_text_from_file or guidance_text_manual
    st.session_state["tw_guidance_text"] = guidance_text
    st.info("尚未提供指引也可先跑（AI 會依一般形式檢核常規）。" if not guidance_text else "已載入指引文字。")

    st.markdown("---")
    st.markdown("### Step 3 – 形式審查 / 完整性檢核（Agent）")

    if not (st.session_state.get("tw_app_effective_md") or "").strip():
        st.warning("尚未產生申請書 Markdown。請先產出草稿。")
        return

    combined_input = f"""=== 申請書草稿（Markdown） ===
{st.session_state.get("tw_app_effective_md","")}

=== 預審 / 形式審查指引（文字/Markdown） ===
{st.session_state.get("tw_guidance_text","") or "（尚未提供指引，請依一般法規常規進行形式檢核）"}
"""

    default_screen_prompt = """你是一位熟悉臺灣「第二、三等級醫療器材查驗登記」的形式審查(預審)審查員。

請根據：
1) 上述申請書草稿
2) 上述預審/形式審查指引（如有）

以「繁體中文 Markdown」輸出預審報告，包含：
- 形式完整性檢核（以表格列出主要文件類別、應附性、申請書是否提及、判定、備註）
- 重要欄位檢核（缺漏/矛盾/需澄清）
- 預審評語摘要（300–600字）
- 不可臆測；無法判斷請明確註記
"""
    agent_run_ui(
        agent_id="tw_screen_review_agent",
        tab_key="tw_screen",
        default_prompt=default_screen_prompt,
        default_input_text=combined_input,
        allow_model_override=True,
        tab_label_for_history="TW Premarket Screen Review",
    )

    st.markdown("---")
    st.markdown("### Step 4 – AI 協助編修申請書內容（Agent）")
    helper_default_prompt = """你是一位協助臺灣醫療器材查驗登記申請人的文件撰寫助手。

請在不改變實際技術與法規內容的前提下：
1) 優化段落結構與標題層級
2) 修正文句語病
3) 資訊不足處以「※待補：...」標註提醒
輸出 Markdown。
"""
    agent_run_ui(
        agent_id="tw_app_doc_helper",
        tab_key="tw_app_helper",
        default_prompt=helper_default_prompt,
        default_input_text=st.session_state.get("tw_app_effective_md", ""),
        allow_model_override=True,
        tab_label_for_history="TW Application Doc Helper",
    )


# =========================================================
# NEW: TFDA 許可證展延（附件/文件）模組
# =========================================================

EXT_SECTIONS = [
    {"id": "S1", "order": 1, "name_zh": "一、醫療器材許可證有效期間展延申請書", "name_en": "License Extension Application Form"},
    {"id": "S2", "order": 2, "name_zh": "二、原許可證", "name_en": "Original License"},
    {"id": "S2b", "order": 3, "name_zh": "標籤、中文核定說明書或包裝核定本", "name_en": "Labeling / IFU / Packaging Approval Copy"},
    {"id": "S3", "order": 4, "name_zh": "三、出產國製售證明", "name_en": "Certificate of Free Sale (CFS)"},
    {"id": "S4", "order": 5, "name_zh": "四、原廠授權登記書", "name_en": "Manufacturer Authorization Letter"},
    {"id": "S5", "order": 6, "name_zh": "五、QMS/QSD 證明文件", "name_en": "QMS/QSD Evidence"},
    {"id": "S6", "order": 7, "name_zh": "六、第一等級查驗登記申請書", "name_en": "Class I Registration Form"},
    {"id": "S7", "order": 8, "name_zh": "七、製造業/販賣業醫療器材商許可執照", "name_en": "Distributor/Manufacturer License"},
    {"id": "S8", "order": 9, "name_zh": "八、委託製造相關核准證明/契約（如適用）", "name_en": "Contract Manufacturing Evidence (if applicable)"},
    {"id": "S9", "order": 10, "name_zh": "九、安全監視或上市後研究計畫報告", "name_en": "PMS / Post-market Study Plan"},
]

EXT_META_REQ = {
    "S3": {"issue_date": True, "reference_case_no": False, "doc_type": True},
    "S4": {"issue_date": True, "reference_case_no": False, "doc_type": True},
    "S5": {"issue_date": True, "reference_case_no": False, "doc_type": True},
}

FILE_STATUS = ["active", "voided", "canceled"]
DOC_TYPE = ["unspecified", "copy", "original"]


def ensure_extension_state():
    if "ext_packet" not in st.session_state:
        # default based on the screenshot text (example): S1,S2,S2b,S3,S4,S5,S7 applicable; S6,S8,S9 not applicable
        packet = {}
        for s in EXT_SECTIONS:
            sid = s["id"]
            default_app = "適用" if sid in ("S1", "S2", "S2b", "S3", "S4", "S5", "S7") else "不適用"
            packet[sid] = {
                "applicability": default_app,  # 適用/不適用/不明
                "self_eval": "廠商自評內容" if sid in ("S3", "S4", "S5", "S8") else "",
                "files": [],  # list of dict metadata
                "checklist": [],  # for S8
            }
        st.session_state["ext_packet"] = packet

    if "ext_guidance_text" not in st.session_state:
        st.session_state["ext_guidance_text"] = ""


def normalize_roc_date_str(roc_str: str) -> str:
    """
    Accept '111/11/10' -> '2022-11-10' (approx rule: ROC year + 1911).
    If already ISO, keep.
    """
    s = (roc_str or "").strip()
    if not s:
        return ""
    if "-" in s and len(s.split("-")[0]) == 4:
        return s
    try:
        parts = s.replace("／", "/").split("/")
        if len(parts) == 3:
            y = int(parts[0]) + 1911
            m = int(parts[1])
            d = int(parts[2])
            return f"{y:04d}-{m:02d}-{d:02d}"
    except Exception:
        pass
    return s


def add_uploaded_files_to_section(section_id: str, uploaded_files: list):
    """
    Convert streamlit UploadedFile(s) into metadata entries under ext_packet[section_id]["files"].
    """
    ensure_extension_state()
    sec = st.session_state["ext_packet"][section_id]
    for uf in uploaded_files or []:
        b = uf.getvalue()
        sec["files"].append({
            "file_name": uf.name,
            "description": "",
            "version_tag": "",
            "doc_type": "unspecified",
            "issue_date_raw": "",
            "issue_date": "",
            "reference_case_no": "",
            "status_flag": "active",
            "notes": "",
            "uploaded_at": datetime.utcnow().isoformat(),
            "sha256": sha256_bytes(b),
            "size_bytes": len(b),
        })


def active_files(sec: dict) -> list:
    return [f for f in sec.get("files", []) if f.get("status_flag") == "active"]


def section_complete(section_id: str, sec: dict) -> (bool, List[str]):
    """
    Determine completeness for applicable sections.
    Rules:
      - if 不適用 => complete
      - if 適用/不明 => need >=1 active file OR justification in self_eval/notes
      - if EXT_META_REQ has requirements => enforce on at least one active file
    """
    app = sec.get("applicability", "不明")
    if app == "不適用":
        return True, []
    reasons = []
    af = active_files(sec)
    if not af and not (sec.get("self_eval", "").strip()):
        reasons.append("未上傳文件且無自評/說明")
        return False, reasons

    req = EXT_META_REQ.get(section_id, {})
    if req and af:
        f0 = af[0]
        if req.get("doc_type") and (f0.get("doc_type") in ("unspecified", "", None)):
            reasons.append("缺少文件型態（正本/影本）")
        if req.get("issue_date") and not (f0.get("issue_date") or "").strip():
            reasons.append("缺少出具日期")
    return (len(reasons) == 0), reasons


def compute_extension_completeness() -> (float, Dict[str, Any]):
    ensure_extension_state()
    packet = st.session_state["ext_packet"]
    total = 0
    ok = 0
    details = {}
    for s in EXT_SECTIONS:
        sid = s["id"]
        sec = packet.get(sid, {})
        app = sec.get("applicability", "不明")
        if app == "不適用":
            details[sid] = {"ok": True, "reasons": [], "app": app}
            continue
        total += 1
        is_ok, reasons = section_complete(sid, sec)
        if is_ok:
            ok += 1
        details[sid] = {"ok": is_ok, "reasons": reasons, "app": app}
    score = (ok / total) if total else 1.0
    return score, details


def extension_packet_to_json() -> dict:
    ensure_extension_state()
    return {
        "packet": st.session_state["ext_packet"],
        "generated_at": datetime.utcnow().isoformat(),
        "note": "本資料為系統整理結果，用於文件完整性管理；非主管機關正式審查意見。",
    }


def build_extension_summary_markdown() -> str:
    ensure_extension_state()
    packet = st.session_state["ext_packet"]
    score, details = compute_extension_completeness()

    lines = []
    lines.append("# TFDA 許可證有效期間展延：附件/文件整包摘要\n")
    lines.append(f"- 整體完整性（適用項目）完成度：**{int(score*100)}%**")
    lines.append(f"- 產生時間（UTC）：{datetime.utcnow().isoformat()}\n")

    lines.append("## A. 附件總表（逐項）\n")
    lines.append("| 序 | 附件項目 | 適用性 | 上傳檔案數(有效) | 狀態 | 缺漏/提醒 |")
    lines.append("|---:|---|---|---:|---|---|")
    for s in sorted(EXT_SECTIONS, key=lambda x: x["order"]):
        sid = s["id"]
        sec = packet[sid]
        app = sec.get("applicability", "不明")
        af_cnt = len(active_files(sec))
        ok = details[sid]["ok"]
        status = "✅ 完整" if ok else ("⚠️ 需補" if app != "不適用" else "—")
        reasons = "；".join(details[sid]["reasons"]) if details[sid]["reasons"] else ""
        lines.append(f"| {s['order']} | {s['name_zh']} | {app} | {af_cnt} | {status} | {reasons} |")

    lines.append("\n## B. 檔案清單（含雜湊）\n")
    for s in sorted(EXT_SECTIONS, key=lambda x: x["order"]):
        sid = s["id"]
        sec = packet[sid]
        if not sec.get("files"):
            continue
        lines.append(f"### {s['name_zh']}")
        for f in sec["files"]:
            lines.append(f"- `{f.get('file_name')}` · status={f.get('status_flag')} · type={f.get('doc_type')} · issue_date={f.get('issue_date') or ''} · sha256={f.get('sha256')[:16]}...")

    if packet.get("S8", {}).get("applicability") == "適用":
        lines.append("\n## C. 委託製造核對清單（若適用）\n")
        lines.append("| 核對項目 | 結果 | 備註 | 證據檔案 |")
        lines.append("|---|---|---|---|")
        for row in packet["S8"].get("checklist", []):
            lines.append(f"| {row.get('item','')} | {row.get('result','')} | {row.get('note','')} | {row.get('evidence','')} |")

    return "\n".join(lines)


def render_tfd_extension_tab():
    ensure_extension_state()
    st.title(t("TFDA Extension"))
    st.markdown(
        """
        <div class="subtle">
        依「許可證有效期間展延」附件清單，建立文件整包管理、完整性指標、並可串接 Agent 產出缺漏分析報告。
        </div>
        """,
        unsafe_allow_html=True,
    )

    # WOW completeness
    score, details = compute_extension_completeness()
    pct = int(score * 100)
    if pct >= 80:
        card_grad = "linear-gradient(135deg,#22c55e,#16a34a)"
        txt = "附件整包完整性良好，適合進行形式審查/送件前自檢。"
    elif pct >= 50:
        card_grad = "linear-gradient(135deg,#f97316,#ea580c)"
        txt = "仍有多項附件或必要欄位待補，建議優先補齊適用項目。"
    else:
        card_grad = "linear-gradient(135deg,#ef4444,#b91c1c)"
        txt = "附件缺漏較多，建議先完成文件整包再進行審查。"

    st.markdown(
        f"""
        <div class="wow-card" style="background:{card_grad};">
          <div class="wow-card-title">EXTENSION PACKET COMPLETENESS</div>
          <div class="wow-card-main">{pct}%</div>
          <div class="subtle" style="margin-top:6px;">{txt}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(score)

    # Missing list
    missing = []
    for s in sorted(EXT_SECTIONS, key=lambda x: x["order"]):
        sid = s["id"]
        if not details[sid]["ok"] and st.session_state["ext_packet"][sid].get("applicability") != "不適用":
            missing.append(f"- {s['name_zh']}：{('；'.join(details[sid]['reasons']) or '需補件')}")

    if missing:
        st.warning("目前需補件/需補欄位：\n" + "\n".join(missing))
    else:
        st.success("所有適用項目均達到基本完整性要求。")

    st.markdown("---")
    st.markdown("## Step 1 – 逐項設定適用性、上傳檔案、填寫中繼資料")

    packet = st.session_state["ext_packet"]

    for s in sorted(EXT_SECTIONS, key=lambda x: x["order"]):
        sid = s["id"]
        sec = packet[sid]

        with st.expander(f"{s['order']}. {s['name_zh']} / {s['name_en']}", expanded=(sid in ("S1", "S3"))):
            colA, colB = st.columns([1, 2])
            with colA:
                sec["applicability"] = st.selectbox(
                    "廠商勾選（適用性）",
                    ["適用", "不適用", "不明"],
                    index=["適用", "不適用", "不明"].index(sec.get("applicability", "不明")),
                    key=f"ext_app_{sid}",
                )
            with colB:
                sec["self_eval"] = st.text_area(
                    "廠商自評內容 / 說明（若適用或有特殊情況請填）",
                    value=sec.get("self_eval", ""),
                    height=80,
                    key=f"ext_self_{sid}",
                )

            up = st.file_uploader(
                "上傳文件（可多檔）",
                type=["pdf", "png", "jpg", "jpeg", "txt", "md", "docx"],
                accept_multiple_files=True,
                key=f"ext_upload_{sid}",
            )
            if st.button("將上傳檔案加入清單", key=f"ext_add_files_{sid}"):
                if up:
                    add_uploaded_files_to_section(sid, up)
                    st.success("已加入文件清單（含 sha256）。")
                else:
                    st.info("尚未選擇檔案。")

            # Editable metadata table
            if sec.get("files"):
                df_files = pd.DataFrame(sec["files"])
                editable_cols = [
                    "file_name", "description", "version_tag", "doc_type",
                    "issue_date_raw", "reference_case_no", "status_flag", "notes",
                    "uploaded_at", "sha256", "size_bytes",
                ]
                df_files = df_files[editable_cols]

                st.caption("可直接編輯：doc_type、issue_date_raw（例如 111/11/10）、reference_case_no、status_flag…")
                edited_df = st.data_editor(
                    df_files,
                    use_container_width=True,
                    num_rows="dynamic",
                    column_config={
                        "doc_type": st.column_config.SelectboxColumn("doc_type", options=DOC_TYPE),
                        "status_flag": st.column_config.SelectboxColumn("status_flag", options=FILE_STATUS),
                    },
                    key=f"ext_files_editor_{sid}",
                )

                # Normalize dates back into sec["files"]
                sec["files"] = edited_df.to_dict(orient="records")
                for f in sec["files"]:
                    raw = (f.get("issue_date_raw") or "").strip()
                    f["issue_date"] = normalize_roc_date_str(raw)

            # Section 8 checklist (only if applicable)
            if sid == "S8" and sec.get("applicability") == "適用":
                st.markdown("### 委託製造核對清單（系統內建）")
                if not sec.get("checklist"):
                    sec["checklist"] = [
                        {"item": "載明委託者及受託製造業者之名稱、地址，且應與原許可證一致。", "result": "不適用", "note": "", "evidence": ""},
                        {"item": "載明之委託製程及醫療器材分級分類品項與原許可證資料相符。", "result": "不適用", "note": "", "evidence": ""},
                        {"item": "應於有效期限內（委託製造契約）。", "result": "不適用", "note": "", "evidence": ""},
                    ]

                evidence_options = [f.get("file_name") for f in sec.get("files", [])] or [""]
                df_chk = pd.DataFrame(sec["checklist"])
                df_chk_edited = st.data_editor(
                    df_chk,
                    use_container_width=True,
                    column_config={
                        "result": st.column_config.SelectboxColumn("結果", options=["符合", "不符合", "不適用"]),
                        "evidence": st.column_config.SelectboxColumn("證據檔案", options=evidence_options),
                    },
                    key="ext_s8_checklist_editor",
                )
                sec["checklist"] = df_chk_edited.to_dict(orient="records")

    st.markdown("---")
    st.markdown("## Step 2 – 輸入展延審查指引（Review Guidance）")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        gfile = st.file_uploader("上傳展延指引 (PDF/TXT/MD)", type=["pdf", "txt", "md"], key="ext_guidance_file")
        gtext = ""
        if gfile is not None:
            suffix = gfile.name.lower().rsplit(".", 1)[-1]
            if suffix == "pdf":
                gtext = extract_pdf_pages_to_text(gfile, 1, 9999)
            else:
                gtext = gfile.read().decode("utf-8", errors="ignore")
    with col_g2:
        gmanual = st.text_area("或貼上展延指引文字/Markdown（繁體中文）", height=180, key="ext_guidance_manual")

    st.session_state["ext_guidance_text"] = gtext or gmanual
    st.info("尚未提供指引亦可先產出摘要，之後再用 Agent 進行對照缺漏分析。" if not st.session_state["ext_guidance_text"] else "已載入展延指引。")

    st.markdown("---")
    st.markdown("## Step 3 – 生成「附件整包摘要 Markdown」並可串接 Agent")

    if st.button("Generate Packet Summary Markdown", key="ext_gen_summary_btn"):
        st.session_state["ext_summary_md"] = build_extension_summary_markdown()
        st.success("已產生摘要 Markdown，可編輯並送到剪貼簿。")

    summary_md = st.session_state.get("ext_summary_md", "")
    if summary_md:
        edited = st.text_area("Packet Summary (Markdown)", value=summary_md, height=260, key="ext_summary_editor")
        st.session_state["ext_summary_md"] = edited
        colx1, colx2, colx3 = st.columns([1, 1, 1])
        with colx1:
            if st.button("Send Summary → Clipboard", key="ext_summary_to_cb"):
                set_clipboard(edited, source="TFDA Extension · Packet Summary")
                st.success("已送到剪貼簿。")
        with colx2:
            st.download_button(
                "Download Summary.md",
                data=edited.encode("utf-8"),
                file_name="tfda_extension_packet_summary.md",
                mime="text/markdown",
            )
        with colx3:
            packet_json = extension_packet_to_json()
            st.download_button(
                "Download Packet.json",
                data=json.dumps(packet_json, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="tfda_extension_packet.json",
                mime="application/json",
            )

    st.markdown("---")
    st.markdown("## Step 4 – Agent：展延附件缺漏/一致性分析（可改 Prompt / Model / Tokens）")

    default_gap_prompt = """你是一位熟悉 TFDA 許可證有效期間展延案件的形式審查員。

請根據：
1) 「附件整包摘要」(Markdown)
2) 「展延審查指引」(如有)

輸出「繁體中文 Markdown」報告，包含：
- 附件完整性缺漏清單（逐項：缺什麼、為什麼重要、建議補法）
- 中繼資料檢核（例如：出具日期、正本/影本、參考案號是否可追溯）
- 一致性/風險提醒（例如：文件狀態作廢/註銷、資訊可能不一致）
- 最後給出「必須補件」與「建議補充」兩個清單

注意：不可臆測不存在的文件內容；若無法判斷請明確寫「依現有輸入無法判斷」。
"""

    combined = f"""=== 附件整包摘要（Markdown）===
{st.session_state.get("ext_summary_md","") or "（尚未產生摘要，請先點擊 Generate）"}

=== 展延審查指引（文字/Markdown）===
{st.session_state.get("ext_guidance_text","") or "（未提供指引，請依一般形式審查常規分析）"}
"""
    agent_run_ui(
        agent_id="tw_license_extension_gap_review_agent",
        tab_key="ext_gap",
        default_prompt=default_gap_prompt,
        default_input_text=combined,
        allow_model_override=True,
        tab_label_for_history="TFDA Extension Gap Review",
    )


# =========================================================
# 510(k) / PDF / Note Keeper / Agents Config (kept; shortened)
# =========================================================

def render_510k_tab():
    st.title(t("510k_tab"))
    col1, col2 = st.columns(2)
    with col1:
        device_name = st.text_input("Device Name")
        k_number = st.text_input("510(k) Number (e.g., K123456)")
    with col2:
        sponsor = st.text_input("Sponsor / Manufacturer (optional)")
        product_code = st.text_input("Product Code (optional)")
    extra_info = st.text_area("Additional context (indications, technology, etc.)")

    default_prompt = f"""
你是 FDA 510(k) 審查助理。請根據使用者輸入，產出「審查導向」的摘要（約 1200–2000 字），並提供多個 Markdown 表格（例如：裝置概述、適應症、性能測試、風險與控制）。
語言：{st.session_state.settings["language"]}。
"""
    combined_input = f"""=== Device Inputs ===
Device name: {device_name}
510(k) number: {k_number}
Sponsor: {sponsor}
Product code: {product_code}

Additional context:
{extra_info}
"""
    agent_run_ui(
        agent_id="fda_510k_intel_agent",
        tab_key="510k",
        default_prompt=default_prompt,
        default_input_text=combined_input,
        tab_label_for_history="510(k) Intelligence",
    )


def render_pdf_to_md_tab():
    st.title(t("PDF → Markdown"))
    uploaded = st.file_uploader("Upload PDF to convert selected pages to Markdown", type=["pdf"], key="pdf_to_md_uploader")
    if uploaded:
        col1, col2 = st.columns(2)
        with col1:
            num_start = st.number_input("From page", min_value=1, value=1, key="pdf_to_md_from")
        with col2:
            num_end = st.number_input("To page", min_value=1, value=5, key="pdf_to_md_to")

        if st.button("Extract Text", key="pdf_to_md_extract_btn"):
            text = extract_pdf_pages_to_text(uploaded, int(num_start), int(num_end))
            st.session_state["pdf_raw_text"] = text

    raw_text = st.session_state.get("pdf_raw_text", "")
    if raw_text:
        default_prompt = f"""
你正在把法規/指引 PDF 擷取文字轉為乾淨 Markdown。

要求：
- 保留標題、清單、表格（盡量用 Markdown table）
- 不得捏造原文沒有的內容
語言：{st.session_state.settings["language"]}。
"""
        agent_run_ui(
            agent_id="pdf_to_markdown_agent",
            tab_key="pdf_to_md",
            default_prompt=default_prompt,
            default_input_text=raw_text,
            tab_label_for_history="PDF → Markdown",
        )
    else:
        st.info("Upload a PDF and click 'Extract Text' to begin.")


def highlight_keywords(text: str, keywords: list[str], color: str) -> str:
    if not text or not keywords:
        return text
    out = text
    for kw in sorted(set([k for k in keywords if k.strip()]), key=len, reverse=True):
        safe_kw = kw.strip()
        if not safe_kw:
            continue
        span = f'<span style="color:{color};font-weight:600;">{safe_kw}</span>'
        out = out.replace(safe_kw, span)
    return out


def render_note_keeper_tab():
    st.title(t("Note Keeper & Magics"))
    st.markdown("### Step 1 – 貼上筆記 → 整理成 Markdown")
    raw_notes = st.text_area("Paste your notes (text or markdown)", height=220, key="notes_raw")

    col_n1, col_n2 = st.columns(2)
    with col_n1:
        note_model = st.selectbox("Model", ALL_MODELS, index=ALL_MODELS.index(st.session_state.settings["model"]), key="note_model")
    with col_n2:
        note_max_tokens = st.number_input("max_tokens", min_value=2000, max_value=120000, value=12000, step=1000, key="note_max_tokens")

    default_note_prompt = """你是一位協助醫療器材/TFDA/510(k) 審查員整理個人筆記的助手。

請將下列筆記整理為：
- 清晰的 Markdown 結構（標題/子標題/條列）
- 保留所有技術與法規重點，不要憑空新增內容
- 標示：關鍵要點、主要風險/疑問、待釐清事項
"""
    note_struct_prompt = st.text_area("Prompt", value=default_note_prompt, height=150, key="note_struct_prompt")

    if st.button("Transform notes", key="note_run_btn"):
        if raw_notes.strip():
            api_keys = st.session_state.get("api_keys", {})
            user_prompt = note_struct_prompt + "\n\n=== RAW NOTES ===\n" + raw_notes
            try:
                out = call_llm(
                    model=note_model,
                    system_prompt="You organize notes into clean markdown.",
                    user_prompt=user_prompt,
                    max_tokens=int(note_max_tokens),
                    temperature=0.15,
                    api_keys=api_keys,
                )
                st.session_state["note_md"] = out
                log_event("Note Keeper", "Note Structurer", note_model, int(len(user_prompt + out) / 4))
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.warning("Please paste notes first.")

    base_note = st.session_state.get("note_md", raw_notes)
    st.text_area("Note (editable)", value=base_note, height=240, key="note_effective")

    st.markdown("---")
    st.markdown("### Magic – AI Keywords（手動標示）")
    kw_input = st.text_input("Keywords (comma-separated)", value="TFDA, QMS, 出產國製售證明, 原廠授權, 作廢", key="kw_input")
    kw_color = st.color_picker("Color", "#ff7f50", key="kw_color")
    if st.button("Apply Keyword Highlighting", key="kw_run_btn"):
        keywords = [k.strip() for k in kw_input.split(",") if k.strip()]
        st.session_state["kw_note"] = highlight_keywords(st.session_state.get("note_effective", ""), keywords, kw_color)

    kw_note = st.session_state.get("kw_note", "")
    if kw_note:
        st.markdown("#### Preview")
        st.markdown(kw_note, unsafe_allow_html=True)


def render_agents_config_tab():
    st.title(t("Agents Config"))
    agents_cfg = st.session_state["agents_cfg"]
    agents_dict = agents_cfg.get("agents", {})

    st.subheader("1. Current Agents Overview")
    if agents_dict:
        df = pd.DataFrame([{
            "agent_id": aid,
            "name": acfg.get("name", ""),
            "model": acfg.get("model", ""),
            "category": acfg.get("category", ""),
        } for aid, acfg in agents_dict.items()])
        st.dataframe(df, use_container_width=True, height=260)
    else:
        st.warning("No agents found in current agents.yaml.")

    st.markdown("---")
    st.subheader("2. Edit Full agents.yaml (raw text)")
    yaml_str_current = yaml.dump(st.session_state["agents_cfg"], allow_unicode=True, sort_keys=False)
    edited_yaml_text = st.text_area("agents.yaml (editable)", value=yaml_str_current, height=320, key="agents_yaml_text_editor")

    col_a1, col_a2, col_a3 = st.columns(3)
    with col_a1:
        if st.button("Apply edited YAML to session", key="apply_edited_yaml"):
            try:
                cfg = yaml.safe_load(edited_yaml_text)
                if not isinstance(cfg, dict) or "agents" not in cfg:
                    st.error("YAML must include top-level 'agents'.")
                else:
                    st.session_state["agents_cfg"] = cfg
                    st.success("Updated agents.yaml in current session.")
            except Exception as e:
                st.error(f"Failed to parse YAML: {e}")

    with col_a2:
        uploaded_agents_tab = st.file_uploader("Upload agents.yaml file", type=["yaml", "yml"], key="agents_yaml_tab_uploader")
        if uploaded_agents_tab is not None:
            try:
                cfg = yaml.safe_load(uploaded_agents_tab.read())
                if "agents" in cfg:
                    st.session_state["agents_cfg"] = cfg
                    st.success("Uploaded agents.yaml applied to this session.")
                else:
                    st.warning("Uploaded file has no 'agents'. Ignoring.")
            except Exception as e:
                st.error(f"Failed to parse uploaded YAML: {e}")

    with col_a3:
        st.download_button(
            "Download current agents.yaml",
            data=yaml_str_current.encode("utf-8"),
            file_name="agents.yaml",
            mime="text/yaml",
            key="download_agents_yaml_current",
        )


# =========================================================
# Mock datasets (3) + mock guidances (3) — Traditional Chinese
# =========================================================

MOCK_CASES: Dict[str, Dict[str, Any]] = {
    "Mock Case A": {
        "application": {
            "doc_no": "衛授食藥字第1130001234號",
            "e_no": "MDE-A-0001",
            "apply_date": "2025-01-20",
            "case_type": "許可證有效期限屆至後六個月內重新申請",
            "device_category": "一般醫材",
            "case_kind": "展延案",
            "origin": "輸入",
            "product_class": "第二等級",
            "similar": "有",
            "replace_flag": "否",
            "prior_app_no": "1120719945 變更案",
            "name_zh": "一次性無菌導尿管",
            "name_en": "Sterile Single-Use Urinary Catheter",
            "indications": "供臨床尿液引流使用，詳如核定中文說明書。",
            "spec_comp": "PVC/矽膠材質，多種尺寸，詳如核定中文說明書。",
            "main_cat": "J.一般醫院及個人使用裝置",
            "item_code": "J.1234",
            "item_name": "導尿管",
            "uniform_id": "24567890",
            "firm_name": "宏遠醫材股份有限公司",
            "firm_addr": "台北市中山區XX路100號10樓",
            "resp_name": "王小明",
            "contact_name": "陳怡君",
            "contact_tel": "02-2345-6789",
            "contact_fax": "02-2345-6790",
            "contact_email": "reg@hongyuan-med.example",
            "confirm_match": True,
            "cert_raps": False,
            "cert_ahwp": True,
            "cert_other": "GDP 內訓證明（2024）",
            "manu_type": "單一製造廠",
            "manu_name": "ACME Medical Devices Inc.",
            "manu_country": "UNITED STATES",
            "manu_addr": "1234 Device Ave, Irvine, CA, USA",
            "manu_note": "最終組裝與滅菌由同廠執行。",
            "auth_applicable": "適用",
            "auth_desc": "原廠授權登記書有效，詳附件。",
            "cfs_applicable": "適用",
            "cfs_desc": "CFS 影本，正本於前案留存。",
            "qms_applicable": "適用",
            "qms_desc": "QMS 證明仍有效。",
            "similar_info": "市場既有同類導尿管產品，差異比較表待補。",
            "labeling_info": "中文說明書與標籤版本一致。",
            "tech_file_info": "結構與材料未變更。",
            "preclinical_info": "無新增測試；引用既有生物相容性與滅菌驗證摘要。",
            "preclinical_replace": "",
            "clinical_just": "不適用",
            "clinical_info": "",
        },
        "extension_guidance": """# 展延案件形式審查指引（Mock A）
## 1. 核心原則
- 展延案重點在於：**許可證資訊一致性**、**文件有效性**、**關鍵證明（CFS/授權/QMS）仍有效且可追溯**。
## 2. 必檢附件（適用時）
1. 許可證有效期間展延申請書：須完整填列並簽章（如系統要求）。
2. 原許可證：須可辨識許可證字號、品名、製造廠、有效期限。
3. 標籤/中文核定說明書/包裝核定本：版本須與許可證系統一致；若更完整須更新系統檔。
4. 出產國製售證明（CFS）：須載明出具日期；影本需可追溯正本留存案號。
5. 原廠授權登記書：須在有效期內；影本需可追溯正本留存案號。
6. QMS/QSD：須在有效期內；需可辨識證書範圍與製造廠一致。
7. 器材商許可執照：名稱/地址/負責人需與申請資料一致。
## 3. 常見缺失
- 影本未註明正本留存資訊、未填出具日期、文件狀態作廢/註銷仍被引用、標籤版本不一致。
"""
    },
    "Mock Case B": {
        "application": {
            "doc_no": "衛授食藥字第1130002222號",
            "e_no": "MDE-B-0002",
            "apply_date": "2025-02-05",
            "case_type": "一般申請案",
            "device_category": "體外診斷器材(IVD)",
            "case_kind": "變更案",
            "origin": "輸入",
            "product_class": "第三等級",
            "similar": "有",
            "replace_flag": "否",
            "prior_app_no": "1110815922",
            "name_zh": "肌酸酐試驗系統",
            "name_en": "Creatinine Assay System",
            "indications": "定量測定人類血清中肌酸酐濃度，用於腎功能評估。",
            "spec_comp": "試劑盒含校正品與控制品，詳如技術檔案。",
            "main_cat": "A.臨床化學及臨床毒理學",
            "item_code": "A.1225",
            "item_name": "肌酸酐試驗系統",
            "uniform_id": "12345678",
            "firm_name": "新星診斷科技有限公司",
            "firm_addr": "新北市板橋區YY路88號8樓",
            "resp_name": "林志強",
            "contact_name": "黃佩珊",
            "contact_tel": "02-8765-4321",
            "contact_fax": "",
            "contact_email": "qa@novadiag.example",
            "confirm_match": True,
            "cert_raps": True,
            "cert_ahwp": False,
            "cert_other": "",
            "manu_type": "單一製造廠",
            "manu_name": "Nova Diagnostics GmbH",
            "manu_country": "EU (Member State)",
            "manu_addr": "Musterstrasse 1, Berlin, Germany",
            "manu_note": "貼標由同廠完成。",
            "auth_applicable": "適用",
            "auth_desc": "授權文件版本更新。",
            "cfs_applicable": "適用",
            "cfs_desc": "CFS 出具日期待補。",
            "qms_applicable": "適用",
            "qms_desc": "ISO 13485 證書附後。",
            "similar_info": "與前案同產品。",
            "labeling_info": "說明書新增警語，待確認是否需變更核定。",
            "tech_file_info": "變更：新增測試方法敘述。",
            "preclinical_info": "分析性能資料補充中。",
            "preclinical_replace": "",
            "clinical_just": "不適用",
            "clinical_info": "",
        },
        "extension_guidance": """# IVD/變更/展延相關文件審查重點（Mock B）
## 必查一致性
- 產品中文/英文名稱、製造廠名稱/地址、器材商資訊必須與原許可證一致（除核准變更項目外）。
## 文件有效性
- CFS / 授權 / QMS 需在有效期內，並能清楚追溯出具日期與證書範圍。
## 標籤/說明書
- 若變更了警語、適應症、使用限制或關鍵性能聲明，需確認是否落入需另案變更或重送核定之範圍。
## 常見缺漏
- 未附出具日期、影本無可追溯資訊、僅上傳作廢版本、系統內說明書未更新。
"""
    },
    "Mock Case C": {
        "application": {
            "doc_no": "衛授食藥字第1130003333號",
            "e_no": "MDE-C-0003",
            "apply_date": "2025-03-01",
            "case_type": "一般申請案",
            "device_category": "一般醫材",
            "case_kind": "展延案",
            "origin": "國產",
            "product_class": "第二等級",
            "similar": "無",
            "replace_flag": "是",
            "prior_app_no": "",
            "name_zh": "低周波治療器",
            "name_en": "Low Frequency Therapy Device",
            "indications": "用於肌肉舒緩與疼痛緩解（非治療性宣稱）。",
            "spec_comp": "主機、電極貼片、導線；多模式輸出。",
            "main_cat": "O.物理醫學科學",
            "item_code": "O.5678",
            "item_name": "電刺激治療器",
            "uniform_id": "87654321",
            "firm_name": "康健醫電股份有限公司",
            "firm_addr": "台中市西屯區ZZ路66號6樓",
            "resp_name": "張雅雯",
            "contact_name": "周承恩",
            "contact_tel": "04-2233-4455",
            "contact_fax": "",
            "contact_email": "reg@healthtron.example",
            "confirm_match": False,
            "cert_raps": False,
            "cert_ahwp": False,
            "cert_other": "",
            "manu_type": "單一製造廠",
            "manu_name": "康健醫電股份有限公司（自製）",
            "manu_country": "TAIWAN， ROC",
            "manu_addr": "台中市西屯區ZZ路66號6樓",
            "manu_note": "自製自售。",
            "auth_applicable": "不適用",
            "auth_desc": "",
            "cfs_applicable": "不適用",
            "cfs_desc": "",
            "qms_applicable": "適用",
            "qms_desc": "QMS 文件待補。",
            "similar_info": "無類似品；需補充市場比較與風險控制。",
            "labeling_info": "中文說明書草稿尚未上傳。",
            "tech_file_info": "技術檔案待補：電氣安全/EMC。",
            "preclinical_info": "替代條款說明待補。",
            "preclinical_replace": "主張引用同系列產品測試，需提供對照與合理性。",
            "clinical_just": "不適用",
            "clinical_info": "",
        },
        "extension_guidance": """# 國產一般醫材展延文件自檢要點（Mock C）
## 1. 必附/常附文件
- 展延申請書、原許可證、標籤/中文核定說明書或包裝核定本
- 器材商許可執照（製造/販賣）
- QMS/QSD（若法規/等級要求）
## 2. 特別注意
- 若申請資料顯示「確認與器材商證照資訊相符」未勾選，通常會被要求補正或澄清。
- 標籤/說明書若缺，屬常見重大缺漏。
## 3. 作廢/註銷
- 上傳文件若為作廢版本，須同時提供有效版本並說明差異與原因。
"""
    },
}


def load_mock_case(case_name: str):
    # Apply TW application mock
    app = MOCK_CASES[case_name]["application"]
    apply_tw_app_dict_to_session(app)
    st.session_state["tw_guidance_text"] = MOCK_CASES[case_name]["extension_guidance"]

    # Also seed TFDA extension packet guidance
    st.session_state["ext_guidance_text"] = MOCK_CASES[case_name]["extension_guidance"]

    # Seed extension packet default + some metadata examples
    ensure_extension_state()
    pkt = st.session_state["ext_packet"]

    # Reset files to empty for clean demo
    for sid in pkt:
        pkt[sid]["files"] = []
        pkt[sid]["checklist"] = []

    # Minimal fake file records (metadata only; no real binaries)
    if case_name == "Mock Case A":
        pkt["S3"]["files"] = [{
            "file_name": "CFS_影本.pdf",
            "description": "出產國製售證明影本",
            "version_tag": "v1",
            "doc_type": "copy",
            "issue_date_raw": "111/11/10",
            "issue_date": normalize_roc_date_str("111/11/10"),
            "reference_case_no": "1110815922",
            "status_flag": "active",
            "notes": "正本在前案留存",
            "uploaded_at": datetime.utcnow().isoformat(),
            "sha256": "demo_cfs_sha256",
            "size_bytes": 123456,
        }]
        pkt["S4"]["files"] = [{
            "file_name": "原廠授權_影本.pdf",
            "description": "原廠授權登記書影本",
            "version_tag": "v2",
            "doc_type": "copy",
            "issue_date_raw": "112/07/27",
            "issue_date": normalize_roc_date_str("112/07/27"),
            "reference_case_no": "1120719945 變更案",
            "status_flag": "active",
            "notes": "正本於變更案留存",
            "uploaded_at": datetime.utcnow().isoformat(),
            "sha256": "demo_auth_sha256",
            "size_bytes": 223344,
        }]
        pkt["S5"]["files"] = [{
            "file_name": "QMS_正本.pdf",
            "description": "QMS/QSD 證明文件",
            "version_tag": "",
            "doc_type": "original",
            "issue_date_raw": "111/10/12",
            "issue_date": normalize_roc_date_str("111/10/12"),
            "reference_case_no": "",
            "status_flag": "active",
            "notes": "",
            "uploaded_at": datetime.utcnow().isoformat(),
            "sha256": "demo_qms_sha256",
            "size_bytes": 998877,
        }]
    if case_name == "Mock Case B":
        pkt["S2b"]["files"] = [{
            "file_name": "中文說明書_核定本.pdf",
            "description": "中文核定說明書（可能含新增警語）",
            "version_tag": "rev3",
            "doc_type": "copy",
            "issue_date_raw": "",
            "issue_date": "",
            "reference_case_no": "",
            "status_flag": "active",
            "notes": "請確認系統內是否需更新",
            "uploaded_at": datetime.utcnow().isoformat(),
            "sha256": "demo_ifu_sha256",
            "size_bytes": 445566,
        }]
        pkt["S3"]["files"] = [{
            "file_name": "CFS.pdf",
            "description": "出產國製售證明",
            "version_tag": "2024",
            "doc_type": "copy",
            "issue_date_raw": "",
            "issue_date": "",
            "reference_case_no": "1110815922",
            "status_flag": "active",
            "notes": "出具日期待補",
            "uploaded_at": datetime.utcnow().isoformat(),
            "sha256": "demo_cfs2_sha256",
            "size_bytes": 111222,
        }]
    if case_name == "Mock Case C":
        pkt["S1"]["files"] = [{
            "file_name": "展延申請書.pdf",
            "description": "許可證有效期間展延申請書",
            "version_tag": "",
            "doc_type": "copy",
            "issue_date_raw": "",
            "issue_date": "",
            "reference_case_no": "",
            "status_flag": "active",
            "notes": "",
            "uploaded_at": datetime.utcnow().isoformat(),
            "sha256": "demo_ext_form_sha256",
            "size_bytes": 222333,
        }]
        pkt["S7"]["files"] = [{
            "file_name": "器材商許可執照.pdf",
            "description": "製造/販賣業許可執照",
            "version_tag": "",
            "doc_type": "copy",
            "issue_date_raw": "",
            "issue_date": "",
            "reference_case_no": "",
            "status_flag": "active",
            "notes": "",
            "uploaded_at": datetime.utcnow().isoformat(),
            "sha256": "demo_license_sha256",
            "size_bytes": 333444,
        }]


# =========================================================
# Main
# =========================================================

st.set_page_config(page_title="Agentic Medical Device Reviewer (WOW)", layout="wide")

if "settings" not in st.session_state:
    st.session_state["settings"] = {
        "theme": "Light",
        "language": "繁體中文",
        "painter_style": "Van Gogh",
        "model": "gpt-4o-mini",
        "max_tokens": 12000,
        "temperature": 0.2,
    }
if "history" not in st.session_state:
    st.session_state["history"] = []

# Load agents.yaml or fallback defaults
if "agents_cfg" not in st.session_state:
    try:
        with open("agents.yaml", "r", encoding="utf-8") as f:
            st.session_state["agents_cfg"] = yaml.safe_load(f)
    except Exception:
        st.session_state["agents_cfg"] = {
            "agents": {
                "fda_510k_intel_agent": {
                    "name": "510(k) Intelligence Agent",
                    "model": "gpt-4o-mini",
                    "system_prompt": "You are an FDA 510(k) analyst.",
                    "max_tokens": 12000,
                    "category": "FDA 510(k)",
                },
                "pdf_to_markdown_agent": {
                    "name": "PDF to Markdown Agent",
                    "model": "gemini-2.5-flash",
                    "system_prompt": "You convert PDF-extracted text into clean markdown.",
                    "max_tokens": 12000,
                    "category": "文件前處理",
                },
                "tw_screen_review_agent": {
                    "name": "TFDA 預審形式審查代理",
                    "model": "gemini-2.5-flash",
                    "system_prompt": "You are a TFDA premarket screen reviewer.",
                    "max_tokens": 12000,
                    "category": "TFDA Premarket",
                },
                "tw_app_doc_helper": {
                    "name": "TFDA 申請書撰寫助手",
                    "model": "gpt-4o-mini",
                    "system_prompt": "You help improve TFDA application documents.",
                    "max_tokens": 12000,
                    "category": "TFDA Premarket",
                },
                "tw_license_extension_gap_review_agent": {
                    "name": "TFDA 展延附件缺漏分析代理",
                    "model": "gemini-2.5-flash",
                    "system_prompt": "You are a TFDA license-extension reviewer focused on completeness and traceability.",
                    "max_tokens": 12000,
                    "category": "TFDA Extension",
                },
            }
        }

render_sidebar()
apply_style(st.session_state.settings["theme"], st.session_state.settings["painter_style"])

tab_labels = [
    t("Dashboard"),
    t("TW Premarket"),
    t("TFDA Extension"),
    t("510k_tab"),
    t("PDF → Markdown"),
    t("Note Keeper & Magics"),
    t("Agents Config"),
]
tabs = st.tabs(tab_labels)

with tabs[0]:
    render_dashboard()
with tabs[1]:
    render_tw_premarket_tab()
with tabs[2]:
    render_tfd_extension_tab()
with tabs[3]:
    render_510k_tab()
with tabs[4]:
    render_pdf_to_md_tab()
with tabs[5]:
    render_note_keeper_tab()
with tabs[6]:
    render_agents_config_tab()
