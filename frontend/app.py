import os
import time
from io import BytesIO

import requests
import streamlit as st
from PIL import Image


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(page_title="RAG Multimodal", layout="wide")
st.title("Asistente RAG Multimodal")
st.caption("Consulta tus manuales PDF con respuestas fundamentadas, citas por p\u00e1gina e im\u00e1genes asociadas.")


def image_fingerprint(image_bytes):
    """Perceptual hash catches identical diagrams exported at different resolutions."""
    try:
        grayscale = Image.open(BytesIO(image_bytes)).convert("L").resize((9, 8))
        pixels = list(grayscale.getdata())
        fingerprint = 0
        for row in range(8):
            for column in range(8):
                fingerprint = (fingerprint << 1) | int(
                    pixels[row * 9 + column] > pixels[row * 9 + column + 1]
                )
        return fingerprint
    except Exception:
        return hash(image_bytes)


def render_source_image(source, seen_images):
    image_url = source.get("image_url")
    if not image_url:
        return
    for previous in seen_images:
        if previous["url"] == image_url:
            st.caption(f"Imagen repetida omitida; ya se mostr\u00f3 en la p\u00e1gina {previous['page']}.")
            return
    try:
        response = requests.get(f"{API_URL}{image_url}", timeout=20)
        response.raise_for_status()
        fingerprint = image_fingerprint(response.content)
        for previous in seen_images:
            if isinstance(fingerprint, int) and isinstance(previous["fingerprint"], int):
                if (fingerprint ^ previous["fingerprint"]).bit_count() <= 3:
                    st.caption(f"Imagen repetida omitida; ya se mostr\u00f3 en la p\u00e1gina {previous['page']}.")
                    return
        seen_images.append({"url": image_url, "fingerprint": fingerprint, "page": source["page_number"]})
        st.image(
            response.content,
            caption=f"Imagen asociada \u00b7 p\u00e1gina {source['page_number']}",
            use_container_width=True,
        )
    except (requests.RequestException, OSError, ValueError) as exc:
        st.warning(f"No se pudo cargar la imagen de la p\u00e1gina {source['page_number']}: {exc}")


with st.sidebar:
    st.header("Documentos")
    uploaded = st.file_uploader("Subir PDF", type=["pdf"])
    if uploaded and st.button("Añadir PDF", type="primary"):
        try:
            response = requests.post(
                f"{API_URL}/documents/upload",
                files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")},
                timeout=60,
            )
            response.raise_for_status()
            job_id = response.json()["job_id"]
            status_box = st.empty()
            while True:
                status_response = requests.get(f"{API_URL}/documents/jobs/{job_id}", timeout=20)
                status_response.raise_for_status()
                job = status_response.json()
                status_box.progress(int(job["progress"]), text=f"{job['status']}: {job['progress']:.0f}%")
                if job["status"] == "COMPLETED":
                    st.success(f"Indexado: {job['result_summary']}")
                    break
                if job["status"] == "FAILED":
                    st.error(job.get("error_message") or "Fall\u00f3 la ingesta")
                    break
                time.sleep(1.5)
        except requests.RequestException as exc:
            st.error(f"No se pudo comunicar con la API: {exc}")

if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("warning"):
            st.warning(message["warning"])
        seen_images = []
        for source in message.get("sources", []):
            st.caption(f"Fuente: {source['filename']} \u00b7 P\u00e1gina {source['page_number']}")
            render_source_image(source, seen_images)

if question := st.chat_input("Haga sus preguntas sobre el documento aqui"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            response = requests.post(
                f"{API_URL}/rag/query", json={"query": question, "top_k": 5}, timeout=120,
            )
            response.raise_for_status()
            result = response.json()
            st.markdown(result["answer"])
            if result.get("warning"):
                st.warning(result["warning"])
            seen_images = []
            for source in result["sources"]:
                st.caption(f"Fuente: {source['filename']} \u00b7 P\u00e1gina {source['page_number']}")
                render_source_image(source, seen_images)
            st.session_state.messages.append({
                "role": "assistant", "content": result["answer"], "sources": result["sources"],
                "warning": result.get("warning"),
            })
        except requests.RequestException as exc:
            st.error(f"No se pudo completar la consulta: {exc}")
