"""
Parcial – NLP y LLMs Avanzado | EAFIT 2026-1
Partes 02, 03 y 04: Laboratorio de Parámetros, Métricas de Similitud y Agente
"""

import json
import re
import time
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="NLP & LLMs Lab – EAFIT",
    page_icon="🧠",
    layout="wide",
)


GEMINI_MODELS = {
    "gemini-2.5-flash": {
        "label": "Gemini 2.5 Flash",
        "input_per_million": 0.30,
        "output_per_million": 2.50,
    },
    "gemini-2.5-flash-lite": {
        "label": "Gemini 2.5 Flash-Lite",
        "input_per_million": 0.10,
        "output_per_million": 0.40,
    },
}

PART3_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "veracidad": {"type": "integer", "minimum": 1, "maximum": 10},
        "coherencia": {"type": "integer", "minimum": 1, "maximum": 10},
        "relevancia": {"type": "integer", "minimum": 1, "maximum": 10},
        "fortalezas": {"type": "string"},
        "debilidades": {"type": "string"},
    },
    "required": [
        "score",
        "veracidad",
        "coherencia",
        "relevancia",
        "fortalezas",
        "debilidades",
    ],
}

PART4_JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "score": {"type": "integer", "minimum": 1, "maximum": 10},
        "veracidad": {"type": "integer", "minimum": 1, "maximum": 10},
        "coherencia": {"type": "integer", "minimum": 1, "maximum": 10},
        "relevancia": {"type": "integer", "minimum": 1, "maximum": 10},
    },
    "required": ["score", "veracidad", "coherencia", "relevancia"],
}

SYSTEM_PROMPT = (
    "Eres MLBot, un tutor experto en Machine Learning y Ciencia de Datos. "
    "Explicas conceptos complejos de forma clara, con ejemplos prácticos y analogías. "
    "Solo respondes preguntas relacionadas con machine learning, estadística, programación "
    "en Python o R, NLP y temas afines. Si la pregunta está fuera de tu dominio, lo dices "
    "amablemente y rediriges la conversación hacia tu especialidad."
)

SECTION_LAB = "■ Laboratorio de Parámetros"
SECTION_METRICS = "■ Métricas de Similitud"
SECTION_AGENT = "■ Agente Especializado"
SECTION_OPTIONS = [SECTION_LAB, SECTION_METRICS, SECTION_AGENT]


def get_secret_value(key: str) -> str:
    """Lee una clave desde st.secrets sin fallar si no existe."""
    try:
        return st.secrets.get(key, "")
    except Exception:
        return ""


def get_client(api_key: str):
    """Crea un cliente del Gemini Developer API con la API key del usuario."""
    from google import genai

    return genai.Client(api_key=api_key)


def get_attr(value: Any, *names: str, default: Any = None) -> Any:
    """Obtiene un atributo o clave probando varias variantes de nombre."""
    if value is None:
        return default

    for name in names:
        if isinstance(value, dict) and name in value:
            return value[name]
        if hasattr(value, name):
            return getattr(value, name)

    return default


def get_response_text(response: Any) -> str:
    """Extrae texto del response de Gemini con fallback a candidates/parts."""
    direct_text = get_attr(response, "text", default="")
    if direct_text:
        return str(direct_text).strip()

    text_parts = []
    for candidate in get_attr(response, "candidates", default=[]) or []:
        content = get_attr(candidate, "content", default=None)
        for part in get_attr(content, "parts", default=[]) or []:
            part_text = get_attr(part, "text", default="")
            if part_text:
                text_parts.append(str(part_text))
    return "\n".join(text_parts).strip()


def parse_json_payload(response: Any) -> dict:
    """Convierte una respuesta JSON del modelo en un diccionario de Python."""
    parsed = get_attr(response, "parsed", default=None)
    if isinstance(parsed, dict):
        return parsed
    if parsed is not None and hasattr(parsed, "model_dump"):
        return parsed.model_dump()
    if isinstance(parsed, str):
        return json.loads(parsed)

    raw_text = get_response_text(response)
    if not raw_text:
        raise ValueError("La respuesta JSON llegó vacía.")

    raw_text = raw_text.strip()
    raw_text = re.sub(r"^```(?:json)?\s*", "", raw_text, flags=re.IGNORECASE)
    raw_text = re.sub(r"\s*```$", "", raw_text)

    try:
        return json.loads(raw_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\{.*\}", raw_text, re.DOTALL)
        if json_match:
            candidate = json_match.group()
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                candidate = re.sub(r"([{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', candidate)
                candidate = candidate.replace("'", '"')
                return json.loads(candidate)
        raise ValueError(f"No se pudo parsear el JSON del judge: {raw_text[:300]}")


def build_generation_config(
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_output_tokens: int,
    frequency_penalty: float,
    presence_penalty: float,
    supports_top_k: bool,
    system_instruction: str | None = None,
    response_json_schema: dict | None = None,
    thinking_budget: int = 0,
):
    """Construye la configuración de generación para Gemini."""
    from google.genai import types

    config_kwargs = {
        "temperature": temperature,
        "top_p": top_p,
        "max_output_tokens": max_output_tokens,
        "frequency_penalty": frequency_penalty,
        "presence_penalty": presence_penalty,
        "thinking_config": types.ThinkingConfig(thinking_budget=thinking_budget),
    }

    if supports_top_k and top_k is not None:
        config_kwargs["top_k"] = int(top_k)
    if system_instruction:
        config_kwargs["system_instruction"] = system_instruction
    if response_json_schema:
        config_kwargs["response_mime_type"] = "application/json"
        config_kwargs["response_json_schema"] = response_json_schema

    return types.GenerateContentConfig(**config_kwargs)


def get_usage_stats(response: Any) -> dict:
    """Extrae uso de tokens del response de Gemini en un formato uniforme."""
    usage = get_attr(response, "usage_metadata", "usageMetadata", default={})
    prompt_tokens = int(get_attr(usage, "prompt_token_count", "promptTokenCount", default=0) or 0)
    completion_tokens = int(
        get_attr(usage, "candidates_token_count", "candidatesTokenCount", default=0) or 0
    )
    thought_tokens = int(
        get_attr(usage, "thoughts_token_count", "thoughtsTokenCount", default=0) or 0
    )
    total_tokens = int(
        get_attr(usage, "total_token_count", "totalTokenCount", default=0)
        or (prompt_tokens + completion_tokens + thought_tokens)
    )
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "thought_tokens": thought_tokens,
        "total_tokens": total_tokens,
    }


def estimate_cost(model_name: str, usage_stats: dict) -> dict:
    """Calcula el costo estimado usando pricing público por millón de tokens."""
    pricing = GEMINI_MODELS[model_name]
    input_cost = usage_stats["prompt_tokens"] * pricing["input_per_million"] / 1_000_000
    output_billable_tokens = usage_stats["completion_tokens"] + usage_stats["thought_tokens"]
    output_cost = output_billable_tokens * pricing["output_per_million"] / 1_000_000
    return {
        "input_cost": input_cost,
        "output_cost": output_cost,
        "total_cost": input_cost + output_cost,
    }


def tokenize_text(text: str) -> list[str]:
    """Tokeniza texto en palabras para BLEU y TTR sin depender de descargas externas."""
    return re.findall(r"\b\w+\b", text.lower(), flags=re.UNICODE)


def compute_ttr(text: str) -> float:
    """Calcula la diversidad léxica Type-Token Ratio."""
    tokens = tokenize_text(text)
    if not tokens:
        return 0.0
    return round(len(set(tokens)) / len(tokens), 3)


@st.cache_resource
def load_sbert_model():
    """Carga el modelo de embeddings una sola vez por sesión."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def fetch_model_capabilities(api_key: str, model_name: str) -> dict:
    """Consulta metadata del modelo para validar soporte de top-k y límites."""
    if not api_key:
        return {
            "verified": False,
            "supports_top_k": True,
            "top_k_default": None,
            "note": "La verificación de top-k se activará cuando ingreses tu API Key.",
        }

    try:
        client = get_client(api_key)
        model_info = client.models.get(model=model_name)
        top_k_default = get_attr(model_info, "top_k", "topK", default=None)
        output_limit = int(get_attr(model_info, "output_token_limit", "outputTokenLimit", default=2048))
        display_name = get_attr(model_info, "display_name", "displayName", default=model_name)
        supports_top_k = top_k_default is not None
        note = (
            f"Metadata verificada para {display_name}. "
            f"Salida máxima del modelo: {output_limit} tokens."
        )
        return {
            "verified": True,
            "supports_top_k": supports_top_k,
            "top_k_default": top_k_default,
            "output_limit": output_limit,
            "note": note,
        }
    except Exception as exc:
        return {
            "verified": False,
            "supports_top_k": True,
            "top_k_default": None,
            "note": f"No se pudo validar la metadata del modelo: {exc}",
        }


def generate_text(
    api_key: str,
    model_name: str,
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int | None,
    max_output_tokens: int,
    frequency_penalty: float,
    presence_penalty: float,
    supports_top_k: bool,
    system_instruction: str | None = None,
):
    """Genera texto con Gemini y devuelve texto + response completo."""
    client = get_client(api_key)
    config = build_generation_config(
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        max_output_tokens=max_output_tokens,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        supports_top_k=supports_top_k,
        system_instruction=system_instruction,
    )
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )
    return get_response_text(response), response


def clamp_score(value: Any, lower: int = 1, upper: int = 10) -> int:
    """Normaliza puntuaciones enteras al rango esperado."""
    value = int(value)
    return max(lower, min(upper, value))


def run_part3_judge(
    api_key: str,
    model_name: str,
    reference_text: str,
    generated_text: str,
    original_prompt: str,
    supports_top_k: bool,
):
    """Ejecuta LLM-as-Judge con salida JSON estructurada para la pestaña de métricas."""
    client = get_client(api_key)
    system_instruction = (
        "Eres un evaluador experto en NLP. Compara la respuesta generada contra la referencia "
        "y devuelve un JSON válido con la evaluación."
    )
    judge_prompt = (
        f"REFERENCIA:\n{reference_text}\n\n"
        f"RESPUESTA GENERADA:\n{generated_text}\n\n"
        f"PROMPT ORIGINAL:\n{original_prompt}\n\n"
        "Evalúa veracidad, coherencia, relevancia y asigna un score general de 1 a 10."
    )
    last_error = None
    for attempt in range(2):
        try:
            config = build_generation_config(
                temperature=0.0,
                top_p=0.1,
                top_k=40,
                max_output_tokens=512,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                supports_top_k=supports_top_k,
                system_instruction=system_instruction,
                response_json_schema=PART3_JUDGE_SCHEMA,
                thinking_budget=0,
            )
            attempt_prompt = judge_prompt
            if attempt == 1:
                attempt_prompt += "\n\nDevuelve solo JSON válido, sin markdown ni texto extra."
            response = client.models.generate_content(
                model=model_name,
                contents=attempt_prompt,
                config=config,
            )
            judge_data = parse_json_payload(response)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise last_error

    judge_data["score"] = clamp_score(judge_data.get("score", 1))
    judge_data["veracidad"] = clamp_score(judge_data.get("veracidad", judge_data["score"]))
    judge_data["coherencia"] = clamp_score(judge_data.get("coherencia", judge_data["score"]))
    judge_data["relevancia"] = clamp_score(judge_data.get("relevancia", judge_data["score"]))
    judge_data["fortalezas"] = str(judge_data.get("fortalezas", "")).strip()
    judge_data["debilidades"] = str(judge_data.get("debilidades", "")).strip()
    return judge_data, response


def run_part4_judge(
    api_key: str,
    model_name: str,
    user_question: str,
    assistant_answer: str,
    supports_top_k: bool,
):
    """Autoevalúa la última respuesta del agente y devuelve score + subtotales."""
    client = get_client(api_key)
    system_instruction = (
        "Eres un evaluador de asistentes conversacionales especializados en ML. "
        "Evalúa la última respuesta según veracidad, coherencia y relevancia y devuelve JSON válido."
    )
    judge_prompt = (
        f"PREGUNTA DEL USUARIO:\n{user_question}\n\n"
        f"RESPUESTA DEL AGENTE:\n{assistant_answer}\n\n"
        "Devuelve una puntuación general de 1 a 10 y los sub-scores."
    )
    last_error = None
    for attempt in range(2):
        try:
            config = build_generation_config(
                temperature=0.0,
                top_p=0.1,
                top_k=40,
                max_output_tokens=256,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                supports_top_k=supports_top_k,
                system_instruction=system_instruction,
                response_json_schema=PART4_JUDGE_SCHEMA,
                thinking_budget=0,
            )
            attempt_prompt = judge_prompt
            if attempt == 1:
                attempt_prompt += "\n\nDevuelve solo JSON válido, sin markdown ni comentarios."
            response = client.models.generate_content(
                model=model_name,
                contents=attempt_prompt,
                config=config,
            )
            judge_data = parse_json_payload(response)
            break
        except Exception as exc:
            last_error = exc
    else:
        raise last_error

    for key in ("score", "veracidad", "coherencia", "relevancia"):
        judge_data[key] = clamp_score(judge_data.get(key, 1))
    return judge_data, response


def history_to_gemini_contents(history: list[dict]) -> list:
    """Convierte el historial guardado en session_state al formato esperado por Gemini."""
    from google.genai import types

    contents = []
    for message in history:
        role = "model" if message["role"] == "assistant" else "user"
        contents.append(
            types.Content(
                role=role,
                parts=[types.Part(text=message["content"])],
            )
        )
    return contents


def build_radar_chart(scores: dict) -> go.Figure:
    """Construye un radar chart con métricas normalizadas a 0-1."""
    categories = list(scores.keys())
    values = list(scores.values())
    normalized = [min(max(value, 0), 1) for value in values]
    return go.Figure(
        go.Scatterpolar(
            r=normalized + [normalized[0]],
            theta=categories + [categories[0]],
            fill="toself",
            name="Métricas",
            line_color="royalblue",
        )
    )


def line_chart(df: pd.DataFrame, y_col: str, title: str, color: str) -> go.Figure:
    """Genera una línea Plotly reutilizable para los históricos del agente."""
    fig = px.line(df, x="Turno", y=y_col, markers=True, title=title)
    fig.update_traces(line_color=color)
    return fig


if "active_section" not in st.session_state:
    st.session_state["active_section"] = SECTION_LAB
if "agent_temp" not in st.session_state:
    st.session_state["agent_temp"] = 0.5
if "agent_max_tok" not in st.session_state:
    st.session_state["agent_max_tok"] = 600

active_section = st.session_state["active_section"]
api_key = get_secret_value("GEMINI_API_KEY")

with st.sidebar:
    st.header("🔑 Gemini API")
    api_key = st.text_input("GEMINI_API_KEY", type="password", value=api_key)
    model_name = st.selectbox(
        "Modelo",
        options=list(GEMINI_MODELS.keys()),
        format_func=lambda item: f"{GEMINI_MODELS[item]['label']} ({item})",
    )

    model_capabilities = fetch_model_capabilities(api_key, model_name)
    supports_top_k = model_capabilities["supports_top_k"]
    max_model_output = int(model_capabilities.get("output_limit", 2048))

    if model_capabilities["verified"]:
        st.success(model_capabilities["note"])
        if supports_top_k:
            default_top_k = model_capabilities.get("top_k_default")
            st.caption(f"Top-k soportado. Valor por defecto del modelo: {default_top_k}.")
        else:
            st.warning("Este modelo no soporta top-k; el control seguirá visible pero deshabilitado.")
    else:
        st.caption(model_capabilities["note"])

    st.caption("Costos estimados según pricing público del Gemini Developer API.")
    max_agent_limit = min(2048, max_model_output)
    st.session_state["agent_max_tok"] = min(int(st.session_state["agent_max_tok"]), max_agent_limit)
    agent_temp = float(st.session_state["agent_temp"])
    agent_max_tok = int(st.session_state["agent_max_tok"])

    st.divider()
    if active_section == SECTION_AGENT:
        st.subheader("🎛️ Agente")
        st.caption("Controles visibles porque estás en la sección del agente conversacional.")
        agent_temp = st.slider("Temperatura agente", 0.0, 2.0, agent_temp, 0.05, key="agent_temp")
        agent_max_tok = st.slider(
            "Max tokens agente",
            50,
            max_agent_limit,
            min(agent_max_tok, max_agent_limit),
            50,
            key="agent_max_tok",
        )
    else:
        st.subheader("🎛️ Agente")
        st.caption("Abre la sección del agente para ver y ajustar sus controles en esta barra lateral.")

st.markdown(
    """
    <style>
    div[role="radiogroup"] {
        gap: 0.5rem;
    }
    div[role="radiogroup"] label {
        border: 1px solid #d7deea;
        border-radius: 0.9rem;
        padding: 0.45rem 0.9rem;
        background: #f7f9fc;
    }
    div[role="radiogroup"] label:has(input:checked) {
        border-color: #4f83ff;
        background: #e8f0ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("### Secciones del Laboratorio")
active_section = st.radio(
    "Secciones",
    options=SECTION_OPTIONS,
    key="active_section",
    horizontal=True,
    label_visibility="collapsed",
)
st.divider()


if active_section == SECTION_LAB:
    st.title("Laboratorio de Sintonización de Parámetros")
    st.caption("Experimenta con los hiperparámetros de generación y observa su efecto.")
    st.caption(
        "Los controles de esta pestaña afectan la prueba individual. "
        "En el experimento comparativo obligatorio, temperatura y top-p se reemplazan "
        "por las 4 configuraciones exigidas por la rúbrica."
    )

    with st.expander("⚙️ Panel de Control Interactivo", expanded=True):
        c1, c2, c3 = st.columns(3)
        with c1:
            temp = st.slider("Temperatura", 0.0, 2.0, 0.7, 0.05, help="Creatividad vs. determinismo")
            top_p = st.slider("Top-p (nucleus)", 0.0, 1.0, 0.9, 0.05, help="Masa de probabilidad acumulada")
        with c2:
            top_k = st.number_input(
                "Top-k",
                min_value=1,
                max_value=100,
                value=40,
                step=1,
                disabled=not supports_top_k,
                help="Vocabulario efectivo por paso de generación",
            )
            max_tok = st.slider(
                "Max tokens",
                50,
                min(2048, max_model_output),
                min(512, max_model_output),
                50,
                help="Longitud máxima de la respuesta generada",
            )
        with c3:
            freq_pen = st.slider(
                "Frequency penalty",
                0.0,
                2.0,
                0.0,
                0.1,
                help="Penalización por repetición de tokens frecuentes",
            )
            pres_pen = st.slider(
                "Presence penalty",
                0.0,
                2.0,
                0.0,
                0.1,
                help="Penalización por aparición previa de tokens",
            )
        st.caption(
            "Nota: top-k y top-p pueden usarse juntos, pero para entender mejor su efecto "
            "conviene variar uno a la vez."
        )

    if not supports_top_k:
        st.info("Top-k no está disponible en este modelo según su metadata; los demás parámetros sí se aplican.")

    prompt_single = st.text_area(
        "Prompt personalizado",
        value="Explica el concepto de atención en transformers.",
        height=90,
    )

    if st.button("Generar respuesta individual", key="btn_single"):
        if not api_key:
            st.error("Ingresa tu API Key de Gemini en la barra lateral.")
        else:
            with st.spinner("Generando respuesta con Gemini…"):
                try:
                    text, response = generate_text(
                        api_key=api_key,
                        model_name=model_name,
                        prompt=prompt_single,
                        temperature=temp,
                        top_p=top_p,
                        top_k=top_k,
                        max_output_tokens=max_tok,
                        frequency_penalty=freq_pen,
                        presence_penalty=pres_pen,
                        supports_top_k=supports_top_k,
                    )
                    usage = get_usage_stats(response)
                    st.markdown("**Respuesta:**")
                    st.write(text)
                    st.caption(
                        "Tokens — entrada: "
                        f"{usage['prompt_tokens']} | salida: {usage['completion_tokens']} | "
                        f"thinking: {usage['thought_tokens']}"
                    )
                except Exception as exc:
                    st.error(f"Error generando respuesta: {exc}")

    st.divider()
    st.subheader("Experimento Comparativo (4 configuraciones)")

    fixed_prompt = "Explica el concepto de atención en transformers."
    configs = [
        {"temp": 0.1, "top_p": 0.9, "label": "T=0.1 / p=0.9"},
        {"temp": 1.5, "top_p": 0.9, "label": "T=1.5 / p=0.9"},
        {"temp": 0.1, "top_p": 0.3, "label": "T=0.1 / p=0.3"},
        {"temp": 1.5, "top_p": 0.3, "label": "T=1.5 / p=0.3"},
    ]
    st.info(f"**Prompt fijo:** {fixed_prompt}")

    if st.button("Ejecutar experimento comparativo", key="btn_compare"):
        if not api_key:
            st.error("Ingresa tu API Key de Gemini en la barra lateral.")
        else:
            results = []
            experiment_context = {
                "model_name": model_name,
                "top_k": int(top_k),
                "max_tok": int(max_tok),
                "freq_pen": float(freq_pen),
                "pres_pen": float(pres_pen),
            }
            progress = st.progress(0, text="Generando respuestas…")
            for index, config in enumerate(configs):
                try:
                    start_time = time.time()
                    text, response = generate_text(
                        api_key=api_key,
                        model_name=model_name,
                        prompt=fixed_prompt,
                        temperature=config["temp"],
                        top_p=config["top_p"],
                        top_k=top_k,
                        max_output_tokens=max_tok,
                        frequency_penalty=freq_pen,
                        presence_penalty=pres_pen,
                        supports_top_k=supports_top_k,
                    )
                    latency = round(time.time() - start_time, 2)
                    visible_tokens = len(tokenize_text(text))
                    results.append(
                        {
                            "label": config["label"],
                            "text": text,
                            "n_tokens": visible_tokens,
                            "ttr": compute_ttr(text),
                            "latency": latency,
                        }
                    )
                except Exception as exc:
                    results.append(
                        {
                            "label": config["label"],
                            "text": f"Error: {exc}",
                            "n_tokens": 0,
                            "ttr": 0,
                            "latency": 0,
                        }
                    )
                progress.progress((index + 1) / len(configs), text=f"Configuración {index + 1}/4…")
            progress.empty()
            st.session_state["compare_results"] = results
            st.session_state["compare_context"] = experiment_context

    if "compare_results" in st.session_state:
        results = st.session_state["compare_results"]
        current_context = {
            "model_name": model_name,
            "top_k": int(top_k),
            "max_tok": int(max_tok),
            "freq_pen": float(freq_pen),
            "pres_pen": float(pres_pen),
        }
        if st.session_state.get("compare_context") != current_context:
            st.warning(
                "Estos resultados fueron generados con otra configuración secundaria. "
                "Si cambiaste top-k, max tokens o penalties, vuelve a ejecutar el experimento."
            )
        cols = st.columns(4)
        for col, result in zip(cols, results):
            with col:
                st.markdown(f"**{result['label']}**")
                st.caption(
                    f"Tokens visibles: {result['n_tokens']} | TTR: {result['ttr']} | "
                    f"Latencia: {result['latency']}s"
                )
                with st.container(border=True):
                    st.write(result["text"])

        df_compare = pd.DataFrame(results)
        fig_tokens = px.bar(
            df_compare,
            x="label",
            y="n_tokens",
            color="label",
            title="Longitud en tokens por configuración",
            labels={"label": "Configuración", "n_tokens": "Tokens visibles de salida"},
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig_tokens.update_layout(showlegend=False)

        fig_ttr = px.bar(
            df_compare,
            x="label",
            y="ttr",
            color="label",
            title="Diversidad léxica (Type-Token Ratio)",
            labels={"label": "Configuración", "ttr": "TTR"},
            color_discrete_sequence=px.colors.qualitative.Safe,
        )
        fig_ttr.update_layout(showlegend=False)

        col_plot_1, col_plot_2 = st.columns(2)
        col_plot_1.plotly_chart(fig_tokens, use_container_width=True)
        col_plot_2.plotly_chart(fig_ttr, use_container_width=True)

        st.subheader("Observaciones del estudiante")
        st.text_area(
            "Documenta aquí el efecto observado de temperatura y top-p sobre las respuestas:",
            height=150,
            key="observations_lab",
            placeholder="Ej: Con T=1.5 las respuestas son más creativas, pero menos precisas.",
        )


if active_section == SECTION_METRICS:
    st.title("Métricas de Similitud y Evaluación Automática")
    st.caption("Compara cuantitativamente un texto de referencia con la salida del LLM.")

    col_ref, col_prompt = st.columns(2)
    with col_ref:
        reference_text = st.text_area(
            "Texto de referencia (ground truth)",
            height=160,
            placeholder="Escribe o pega aquí la respuesta esperada…",
        )
    with col_prompt:
        eval_prompt = st.text_area(
            "Prompt enviado al LLM",
            value="Explica el mecanismo de atención (attention) en los modelos Transformer.",
            height=160,
        )

    bert_lang = st.selectbox(
        "Idioma para BERTScore",
        ["es", "en"],
        index=0,
        help="Selecciona el idioma dominante del texto de referencia y de la respuesta generada.",
    )

    if st.button("Evaluar", key="btn_eval"):
        if not api_key:
            st.error("Ingresa tu API Key de Gemini en la barra lateral.")
        elif not reference_text.strip():
            st.error("Ingresa un texto de referencia.")
        else:
            try:
                with st.spinner("Generando respuesta candidata…"):
                    generated_text, response = generate_text(
                        api_key=api_key,
                        model_name=model_name,
                        prompt=eval_prompt,
                        temperature=0.3,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=min(768, max_model_output),
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        supports_top_k=supports_top_k,
                    )
                st.session_state["eval_generated"] = generated_text
                st.session_state["eval_usage"] = get_usage_stats(response)
                st.session_state["eval_prompt_used"] = eval_prompt
                st.session_state["eval_reference"] = reference_text
                st.session_state["eval_bert_lang"] = bert_lang

                scores = {}

                with st.spinner("Calculando Similitud Coseno…"):
                    from sklearn.metrics.pairwise import cosine_similarity

                    sbert = load_sbert_model()
                    embeddings = sbert.encode([reference_text, generated_text])
                    cosine_value = float(cosine_similarity([embeddings[0]], [embeddings[1]])[0][0])
                    scores["Cosine"] = round(cosine_value, 4)

                with st.spinner("Calculando BLEU…"):
                    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

                    reference_tokens = tokenize_text(reference_text)
                    hypothesis_tokens = tokenize_text(generated_text)
                    smoothing = SmoothingFunction().method1
                    bleu_value = sentence_bleu(
                        [reference_tokens],
                        hypothesis_tokens,
                        smoothing_function=smoothing,
                    )
                    scores["BLEU"] = round(float(bleu_value), 4)

                with st.spinner("Calculando ROUGE-L…"):
                    from rouge_score import rouge_scorer

                    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
                    rouge_value = scorer.score(reference_text, generated_text)["rougeL"].fmeasure
                    scores["ROUGE-L"] = round(float(rouge_value), 4)

                with st.spinner("Calculando BERTScore…"):
                    from bert_score import score as bert_score_fn

                    _, _, bert_f1 = bert_score_fn(
                        [generated_text],
                        [reference_text],
                        lang=bert_lang,
                        verbose=False,
                    )
                    scores["BERTScore"] = round(float(bert_f1[0]), 4)

                with st.spinner("Ejecutando LLM-as-Judge…"):
                    judge_json, judge_response = run_part3_judge(
                        api_key=api_key,
                        model_name=model_name,
                        reference_text=reference_text,
                        generated_text=generated_text,
                        original_prompt=eval_prompt,
                        supports_top_k=supports_top_k,
                    )
                    scores["LLM-Judge"] = round(judge_json["score"] / 10, 4)
                    st.session_state["judge_json"] = judge_json
                    st.session_state["judge_usage"] = get_usage_stats(judge_response)

                st.session_state["eval_scores"] = scores
            except Exception as exc:
                st.error(f"Error durante la evaluación: {exc}")

    if "eval_generated" in st.session_state:
        st.markdown("**Respuesta generada:**")
        st.write(st.session_state["eval_generated"])

    if "eval_scores" in st.session_state:
        scores = st.session_state["eval_scores"]
        st.divider()
        st.subheader("Resultados de las métricas")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Cosine Similarity", scores.get("Cosine", "N/A"))
        m2.metric("BLEU", scores.get("BLEU", "N/A"))
        m3.metric("ROUGE-L", scores.get("ROUGE-L", "N/A"))
        m4.metric("BERTScore F1", scores.get("BERTScore", "N/A"))
        judge_metric = scores.get("LLM-Judge")
        m5.metric("LLM-Judge (/10)", round(judge_metric * 10, 1) if judge_metric is not None else "N/A")

        if "judge_json" in st.session_state:
            judge_json = st.session_state["judge_json"]
            with st.expander("Detalle LLM-as-Judge"):
                j1, j2, j3 = st.columns(3)
                j1.metric("Veracidad", judge_json.get("veracidad"))
                j2.metric("Coherencia", judge_json.get("coherencia"))
                j3.metric("Relevancia", judge_json.get("relevancia"))
                st.markdown(f"**Fortalezas:** {judge_json.get('fortalezas', '')}")
                st.markdown(f"**Debilidades:** {judge_json.get('debilidades', '')}")

        valid_scores = {name: value for name, value in scores.items() if value is not None}
        if valid_scores:
            fig_radar = build_radar_chart(valid_scores)
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                title="Radar de Métricas (normalizadas 0–1)",
                showlegend=False,
            )
            st.plotly_chart(fig_radar, use_container_width=True)


if active_section == SECTION_AGENT:
    st.title("Agente Especializado – Tutor de Machine Learning")
    st.caption("Chat con memoria conversacional y métricas de producción en tiempo real.")

    if "agent_history" not in st.session_state:
        st.session_state["agent_history"] = []
    if "agent_metrics" not in st.session_state:
        st.session_state["agent_metrics"] = []

    if st.button("🗑 Limpiar conversación"):
        st.session_state["agent_history"] = []
        st.session_state["agent_metrics"] = []
        st.rerun()

    for message in st.session_state["agent_history"]:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    user_input = st.chat_input("Escribe tu pregunta sobre Machine Learning…")

    if user_input:
        if not api_key:
            st.error("Ingresa tu API Key de Gemini en la barra lateral.")
        else:
            try:
                with st.spinner("MLBot está pensando…"):
                    client = get_client(api_key)
                    chat_config = build_generation_config(
                        temperature=agent_temp,
                        top_p=0.9,
                        top_k=40,
                        max_output_tokens=agent_max_tok,
                        frequency_penalty=0.0,
                        presence_penalty=0.0,
                        supports_top_k=supports_top_k,
                        system_instruction=SYSTEM_PROMPT,
                    )
                    chat = client.chats.create(
                        model=model_name,
                        history=history_to_gemini_contents(st.session_state["agent_history"]),
                        config=chat_config,
                    )

                    start_time = time.time()
                    response = chat.send_message(user_input)
                    latency = time.time() - start_time
                    response_text = get_response_text(response)

                    main_usage = get_usage_stats(response)
                    main_cost = estimate_cost(model_name, main_usage)
                    tps = main_usage["completion_tokens"] / latency if latency > 0 else 0.0

                    judge_details, judge_response = run_part4_judge(
                        api_key=api_key,
                        model_name=model_name,
                        user_question=user_input,
                        assistant_answer=response_text,
                        supports_top_k=supports_top_k,
                    )
                    judge_score = judge_details["score"]
                    judge_usage = get_usage_stats(judge_response)
                    judge_cost_total = estimate_cost(model_name, judge_usage)["total_cost"]
                    total_turn_cost = main_cost["total_cost"] + judge_cost_total

                st.session_state["agent_history"].append({"role": "user", "content": user_input})
                st.session_state["agent_history"].append({"role": "assistant", "content": response_text})

                turn_number = len(st.session_state["agent_metrics"]) + 1
                st.session_state["agent_metrics"].append(
                    {
                        "Turno": turn_number,
                        "Latencia (s)": round(latency, 2),
                        "TPS": round(tps, 2),
                        "Tokens entrada": main_usage["prompt_tokens"],
                        "Tokens salida": main_usage["completion_tokens"],
                        "Thinking tokens": main_usage["thought_tokens"],
                        "Costo respuesta USD": round(main_cost["total_cost"], 6),
                        "Costo judge USD": round(judge_cost_total, 6),
                        "Costo total USD": round(total_turn_cost, 6),
                        "LLM-Judge": judge_score,
                        "Judge veracidad": judge_details["veracidad"],
                        "Judge coherencia": judge_details["coherencia"],
                        "Judge relevancia": judge_details["relevancia"],
                    }
                )
                st.rerun()
            except Exception as exc:
                st.error(f"Error en el agente: {exc}")

    if st.session_state["agent_metrics"]:
        st.divider()
        st.subheader("Historial de Métricas por Turno")
        df_metrics = pd.DataFrame(st.session_state["agent_metrics"])
        df_metrics["Costo acumulado USD"] = df_metrics["Costo total USD"].cumsum()

        latest_turn = df_metrics.iloc[-1]
        with st.expander("📊 Métricas del último turno", expanded=True):
            row1_col1, row1_col2, row1_col3 = st.columns(3)
            row1_col1.metric("Latencia", f"{latest_turn['Latencia (s)']:.2f}s")
            row1_col2.metric("TPS", f"{latest_turn['TPS']:.2f}")
            row1_col3.metric(
                "LLM-Judge",
                f"{int(latest_turn['LLM-Judge'])}/10" if pd.notna(latest_turn["LLM-Judge"]) else "N/A",
            )

            row2_col1, row2_col2, row2_col3 = st.columns(3)
            row2_col1.metric("Tokens entrada", int(latest_turn["Tokens entrada"]))
            row2_col2.metric("Tokens salida", int(latest_turn["Tokens salida"]))
            row2_col3.metric("Costo total USD", f"${latest_turn['Costo total USD']:.6f}")

            st.caption(
                "Judge detalle — "
                f"Veracidad: {int(latest_turn['Judge veracidad'])} | "
                f"Coherencia: {int(latest_turn['Judge coherencia'])} | "
                f"Relevancia: {int(latest_turn['Judge relevancia'])}"
            )

        tab_lat, tab_tps, tab_judge, tab_cost = st.tabs(["Latencia", "TPS", "LLM-Judge", "Costo"])

        tab_lat.plotly_chart(
            line_chart(df_metrics, "Latencia (s)", "Latencia por turno (s)", "tomato"),
            use_container_width=True,
        )
        tab_tps.plotly_chart(
            line_chart(df_metrics, "TPS", "Tokens por segundo", "green"),
            use_container_width=True,
        )
        if df_metrics["LLM-Judge"].notna().any():
            tab_judge.plotly_chart(
                line_chart(
                    df_metrics.dropna(subset=["LLM-Judge"]),
                    "LLM-Judge",
                    "Puntuación LLM-Judge (1-10)",
                    "goldenrod",
                ),
                use_container_width=True,
            )

        cost_fig = go.Figure()
        cost_fig.add_trace(
            go.Scatter(
                x=df_metrics["Turno"],
                y=df_metrics["Costo total USD"],
                mode="lines+markers",
                name="Costo por turno",
                line=dict(color="royalblue"),
            )
        )
        cost_fig.add_trace(
            go.Scatter(
                x=df_metrics["Turno"],
                y=df_metrics["Costo acumulado USD"],
                mode="lines+markers",
                name="Costo acumulado",
                line=dict(color="purple"),
            )
        )
        cost_fig.update_layout(title="Costo por turno y costo acumulado (USD)", xaxis_title="Turno")
        tab_cost.plotly_chart(cost_fig, use_container_width=True)

        with st.expander("Ver tabla completa de métricas"):
            st.dataframe(df_metrics, use_container_width=True)
