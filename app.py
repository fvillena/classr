"""
classr — entrena un clasificador de texto ligero desde el navegador y lo expone
como servicio web (REST) para usarlo, por ejemplo, desde n8n.

Ejecutar:
    python app.py                     # http://localhost:7860
    PUBLIC_URL=https://mi.dominio python app.py

Rutas:
    /                                  interfaz Gradio
    POST /api/models/{model_id}/predict   {"text": "..."} o {"texts": ["...", "..."]}
    GET  /api/models/{model_id}/predict?text=...
    GET  /api/models/{model_id}           información y métricas del modelo
    GET  /docs                             documentación OpenAPI (Swagger)
"""

from __future__ import annotations

import json
import os
import re
import secrets
import threading
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import gradio as gr
import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

# --------------------------------------------------------------------------- #
# Configuración
# --------------------------------------------------------------------------- #
MODELS_DIR = Path(os.environ.get("MODELS_DIR", Path(__file__).parent / "models"))
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PUBLIC_URL = os.environ.get("PUBLIC_URL", "").rstrip("/")
MAX_ROWS = int(os.environ.get("MAX_ROWS", 200_000))
MAX_BATCH = 1000
MODEL_ID_RE = re.compile(r"^[A-Za-z0-9_-]{4,32}$")

ALGORITHMS = {
    "Regresión logística": "logreg",
    "Naive Bayes (Complement)": "nb",
}
FEATURES = {
    "Palabras (1-2 gramas)": "word",
    "Caracteres (2-5 gramas)": "char",
}

# Caché LRU de modelos en memoria: guarda los CACHE_SIZE usados más recientemente;
# el resto se vuelve a leer desde disco cuando se piden.
CACHE_SIZE = max(1, int(os.environ.get("CACHE_SIZE", 32)))
_cache: OrderedDict[str, Pipeline] = OrderedDict()
_cache_lock = threading.Lock()


def _cache_put(model_id: str, pipe: Pipeline) -> None:
    with _cache_lock:
        _cache[model_id] = pipe
        _cache.move_to_end(model_id)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)  # descarta el menos usado recientemente


def _cache_get(model_id: str) -> Pipeline | None:
    with _cache_lock:
        pipe = _cache.get(model_id)
        if pipe is not None:
            _cache.move_to_end(model_id)
        return pipe


# --------------------------------------------------------------------------- #
# Persistencia de modelos
# --------------------------------------------------------------------------- #
def _paths(model_id: str) -> tuple[Path, Path]:
    return MODELS_DIR / f"{model_id}.joblib", MODELS_DIR / f"{model_id}.json"


def save_model(model_id: str, pipe: Pipeline, meta: dict) -> None:
    model_path, meta_path = _paths(model_id)
    joblib.dump(pipe, model_path)
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    _cache_put(model_id, pipe)


def load_model(model_id: str) -> Pipeline:
    if not MODEL_ID_RE.match(model_id or ""):
        raise KeyError(model_id)
    pipe = _cache_get(model_id)
    if pipe is not None:
        return pipe
    model_path, _ = _paths(model_id)
    if not model_path.exists():
        raise KeyError(model_id)
    pipe = joblib.load(model_path)
    _cache_put(model_id, pipe)
    return pipe


def load_meta(model_id: str) -> dict:
    if not MODEL_ID_RE.match(model_id or ""):
        raise KeyError(model_id)
    _, meta_path = _paths(model_id)
    if not meta_path.exists():
        raise KeyError(model_id)
    return json.loads(meta_path.read_text(encoding="utf-8"))


def predict_texts(pipe: Pipeline, texts: list[str]) -> list[dict]:
    texts = [str(t) for t in texts]
    probs = pipe.predict_proba(texts)
    classes = [str(c) for c in pipe.classes_]
    out = []
    for row in probs:
        best = int(np.argmax(row))
        out.append(
            {
                "label": classes[best],
                "confidence": round(float(row[best]), 4),
                "probabilities": {c: round(float(p), 4) for c, p in zip(classes, row)},
            }
        )
    return out


# --------------------------------------------------------------------------- #
# Lectura del dataset
# --------------------------------------------------------------------------- #
def read_dataset(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix in (".csv", ".txt"):
        # sep=None detecta automáticamente , ; o tabulador
        df = pd.read_csv(path, sep=None, engine="python", encoding_errors="replace")
    elif suffix == ".tsv":
        df = pd.read_csv(path, sep="\t", encoding_errors="replace")
    elif suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif suffix == ".json":
        df = pd.read_json(path)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    else:
        raise gr.Error(f"Formato no soportado: {suffix}. Usa CSV, TSV, JSON, JSONL o XLSX.")
    df.columns = [str(c) for c in df.columns]
    return df


def guess_columns(df: pd.DataFrame) -> tuple[str, str]:
    cols = list(df.columns)
    lower = {c.lower(): c for c in cols}
    text = next((lower[k] for k in ("text", "texto", "sentence", "review", "content", "mensaje", "tweet") if k in lower), None)
    label = next((lower[k] for k in ("label", "etiqueta", "clase", "class", "category", "categoria", "sentiment", "target") if k in lower), None)
    if text is None:
        # la columna de texto suele ser la de strings más largos
        obj = [c for c in cols if df[c].dtype == object]
        text = max(obj or cols, key=lambda c: df[c].astype(str).str.len().mean())
    if label is None:
        others = [c for c in cols if c != text]
        label = min(others, key=lambda c: df[c].nunique()) if others else text
    return text, label


# --------------------------------------------------------------------------- #
# Entrenamiento
# --------------------------------------------------------------------------- #
def build_pipeline(algorithm: str, features: str) -> Pipeline:
    if features == "char":
        vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2,
                              max_features=100_000, sublinear_tf=True, lowercase=True)
    else:
        vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=1,
                              max_features=100_000, sublinear_tf=True, lowercase=True,
                              strip_accents=None)
    if algorithm == "nb":
        clf = ComplementNB(alpha=0.5)
    else:
        clf = LogisticRegression(max_iter=2000, C=4.0, class_weight="balanced")
    return Pipeline([("tfidf", vec), ("clf", clf)])


def top_features(pipe: Pipeline, n: int = 10) -> pd.DataFrame:
    vec: TfidfVectorizer = pipe.named_steps["tfidf"]
    clf = pipe.named_steps["clf"]
    names = vec.get_feature_names_out()
    classes = [str(c) for c in clf.classes_]
    if hasattr(clf, "coef_"):
        weights = clf.coef_
        if weights.shape[0] == 1:  # binario: una sola fila de coeficientes
            weights = np.vstack([-weights[0], weights[0]])
    else:  # ComplementNB: feature_log_prob_ menor = más característico de la clase
        weights = -clf.feature_log_prob_
    rows = {}
    for i, c in enumerate(classes):
        idx = np.argsort(weights[i])[::-1][:n]
        rows[c] = [names[j] for j in idx]
    return pd.DataFrame(rows)


def base_url(request: gr.Request | None) -> str:
    if PUBLIC_URL:
        return PUBLIC_URL
    if request is not None:
        h = {k.lower(): v for k, v in request.headers.items()}
        host = h.get("x-forwarded-host") or h.get("host")
        proto = h.get("x-forwarded-proto") or ("https" if h.get("x-forwarded-ssl") == "on" else "http")
        if host:
            return f"{proto}://{host}"
    return "http://localhost:7860"


def endpoint_markdown(model_id: str, request: gr.Request | None) -> str:
    url = f"{base_url(request)}/api/models/{model_id}/predict"
    return f"""### Tu modelo: `{model_id}`

**URL del servicio (POST):**
```
{url}
```

**Cuerpo JSON:** `{{"text": "tu texto aquí"}}` — o varios a la vez: `{{"texts": ["uno", "dos"]}}`

**Respuesta:**
```json
{{"model_id": "{model_id}", "label": "...", "confidence": 0.93, "probabilities": {{"...": 0.93, "...": 0.07}}}}
```

**Probar con curl:**
```bash
curl -X POST {url} -H "Content-Type: application/json" -d '{{"text": "me encantó"}}'
```

**En n8n** → nodo *HTTP Request*:
- Method: `POST`
- URL: `{url}`
- Body Content Type: `JSON` → Specify Body: *Using JSON* → `{{{{ {{ "text": $json.text }} }}}}`
- La etiqueta queda en `{{{{ $json.label }}}}` para el siguiente nodo (p. ej. un *Switch*).

También funciona por GET para pruebas rápidas: `{url}?text=hola`
"""


def on_upload(file):
    if file is None:
        return (gr.update(choices=[], value=None), gr.update(choices=[], value=None),
                gr.update(value=None, visible=False), "")
    df = read_dataset(file if isinstance(file, str) else file.name)
    cols = list(df.columns)
    text, label = guess_columns(df)
    info = f"**{len(df):,} filas**, {len(cols)} columnas. Revisa que las columnas elegidas sean correctas."
    return (gr.update(choices=cols, value=text), gr.update(choices=cols, value=label),
            gr.update(value=df.head(10), visible=True), info)


def train(file, text_col, label_col, algorithm_name, features_name, test_size, name,
          request: gr.Request, progress=gr.Progress()):
    if file is None:
        raise gr.Error("Primero sube un dataset.")
    if not text_col or not label_col or text_col == label_col:
        raise gr.Error("Elige una columna de texto y una de etiqueta distintas.")

    progress(0.1, desc="Leyendo dataset")
    df = read_dataset(file if isinstance(file, str) else file.name)
    df = df[[text_col, label_col]].dropna()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[(df[text_col] != "") & (df[label_col] != "")]
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=42)

    counts = df[label_col].value_counts()
    if len(counts) < 2:
        raise gr.Error("Se necesitan al menos 2 clases distintas en la columna de etiqueta.")
    if len(counts) > 200:
        raise gr.Error(f"La columna '{label_col}' tiene {len(counts)} valores distintos: "
                       "¿seguro que es la etiqueta y no el texto?")
    if len(df) < 10:
        raise gr.Error("El dataset es demasiado pequeño (menos de 10 ejemplos válidos).")

    X, y = df[text_col].tolist(), df[label_col].tolist()
    stratify = y if counts.min() >= 2 else None
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=float(test_size),
                                              random_state=42, stratify=stratify)

    algorithm, features = ALGORITHMS[algorithm_name], FEATURES[features_name]
    progress(0.4, desc="Entrenando")
    pipe = build_pipeline(algorithm, features)
    try:
        pipe.fit(X_tr, y_tr)
    except ValueError as e:  # p. ej. vocabulario vacío con min_df=2 en datasets mínimos
        raise gr.Error(f"No se pudo entrenar: {e}")

    progress(0.75, desc="Evaluando")
    y_pred = pipe.predict(X_te)
    labels = [str(c) for c in pipe.classes_]
    acc = accuracy_score(y_te, y_pred)
    f1 = f1_score(y_te, y_pred, average="macro")
    report = classification_report(y_te, y_pred, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(
        [{"clase": c, "precision": round(report[c]["precision"], 3), "recall": round(report[c]["recall"], 3),
          "f1": round(report[c]["f1-score"], 3), "soporte": int(report[c]["support"])} for c in labels]
    )
    cm = confusion_matrix(y_te, y_pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"real: {c}" for c in labels], columns=[f"pred: {c}" for c in labels])
    cm_df.insert(0, "", cm_df.index)

    # Reentrena con todos los datos para el modelo que se publica
    progress(0.9, desc="Reentrenando con todos los datos")
    final = build_pipeline(algorithm, features).fit(X, y)

    model_id = secrets.token_urlsafe(6).replace("_", "x").replace("-", "y")
    meta = {
        "model_id": model_id,
        "name": (name or "").strip() or None,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "algorithm": algorithm_name,
        "features": features_name,
        "classes": labels,
        "n_examples": len(df),
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "metrics": {"accuracy": round(acc, 4), "f1_macro": round(f1, 4)},
        "class_distribution": {str(k): int(v) for k, v in counts.items()},
    }
    save_model(model_id, final, meta)

    summary = (f"## ✅ Modelo entrenado\n\n"
               f"| Accuracy | F1 macro | Ejemplos (train / test) | Clases |\n|---|---|---|---|\n"
               f"| **{acc:.3f}** | **{f1:.3f}** | {len(X_tr):,} / {len(X_te):,} | {len(labels)} |\n\n"
               f"Las métricas se calculan sobre el {int(float(test_size) * 100)}% de prueba; "
               f"el modelo publicado se reentrenó con el 100% de los datos.")
    return (model_id, summary, gr.update(value=report_df, visible=True),
            gr.update(value=cm_df, visible=True), gr.update(value=top_features(final), visible=True),
            endpoint_markdown(model_id, request), model_id,
            f"Modelo **{model_id}** listo. Escribe un texto para probarlo.")


def try_model(model_id: str, text: str):
    model_id = (model_id or "").strip()
    if not model_id:
        raise gr.Error("Entrena un modelo o escribe el ID de uno existente.")
    if not (text or "").strip():
        raise gr.Error("Escribe un texto para clasificar.")
    try:
        pipe = load_model(model_id)
    except KeyError:
        raise gr.Error(f"No existe el modelo '{model_id}'.")
    res = predict_texts(pipe, [text])[0]
    return {k: v for k, v in res["probabilities"].items()}


def load_existing(model_id: str, request: gr.Request):
    model_id = (model_id or "").strip()
    try:
        meta = load_meta(model_id)
    except KeyError:
        raise gr.Error(f"No existe el modelo '{model_id}'.")
    m = meta["metrics"]
    info = (f"Modelo **{model_id}**{' — ' + meta['name'] if meta.get('name') else ''}: "
            f"clases {', '.join(meta['classes'])} · accuracy {m['accuracy']} · F1 macro {m['f1_macro']}")
    return model_id, info, endpoint_markdown(model_id, request)


# --------------------------------------------------------------------------- #
# Interfaz Gradio
# --------------------------------------------------------------------------- #
with gr.Blocks(title="classr · clasificador de texto") as demo:
    current_id = gr.State("")
    gr.Markdown("# classr — entrena tu clasificador de texto\n"
                "Sube un dataset con una columna de **texto** y una de **etiqueta**, entrena un modelo "
                "TF-IDF + clasificador lineal y úsalo desde aquí o como servicio web (n8n, curl, Python…).")

    with gr.Tab("1 · Entrenar"):
        with gr.Row():
            with gr.Column(scale=1):
                file = gr.File(label="Dataset (CSV, TSV, JSON, JSONL o XLSX)",
                               file_types=[".csv", ".tsv", ".txt", ".json", ".jsonl", ".xlsx", ".xls"])
                info = gr.Markdown()
                text_col = gr.Dropdown(label="Columna de texto", choices=[])
                label_col = gr.Dropdown(label="Columna de etiqueta", choices=[])
                with gr.Accordion("Opciones", open=False):
                    algorithm = gr.Radio(list(ALGORITHMS), value="Regresión logística", label="Algoritmo")
                    features = gr.Radio(list(FEATURES), value="Palabras (1-2 gramas)", label="Representación")
                    test_size = gr.Slider(0.1, 0.5, value=0.2, step=0.05, label="Proporción de prueba")
                    name = gr.Textbox(label="Nombre del modelo (opcional)", placeholder="sentimiento-tweets")
                train_btn = gr.Button("Entrenar", variant="primary")
            with gr.Column(scale=2):
                preview = gr.Dataframe(label="Vista previa", visible=False, wrap=True)
                summary = gr.Markdown()
                report_df = gr.Dataframe(label="Métricas por clase", visible=False)
                cm_df = gr.Dataframe(label="Matriz de confusión (filas = real, columnas = predicho)", visible=False)
                feats_df = gr.Dataframe(label="Términos más característicos de cada clase", wrap=True, visible=False)

    with gr.Tab("2 · Probar"):
        with gr.Row():
            model_box = gr.Textbox(label="ID del modelo", info="Se completa al entrenar; también puedes pegar uno anterior.")
            load_btn = gr.Button("Cargar modelo", scale=0)
        model_info = gr.Markdown()
        text_in = gr.Textbox(label="Escribe un texto", lines=3, placeholder="Este producto es excelente…")
        go_btn = gr.Button("Clasificar", variant="primary")
        probs_out = gr.Label(label="Predicción", num_top_classes=10)

    with gr.Tab("3 · Servicio web / n8n"):
        endpoint_md = gr.Markdown("Entrena un modelo para obtener su URL.")

    file.change(on_upload, file, [text_col, label_col, preview, info])
    train_btn.click(train, [file, text_col, label_col, algorithm, features, test_size, name],
                    [model_box, summary, report_df, cm_df, feats_df, endpoint_md, current_id, model_info])
    go_btn.click(try_model, [model_box, text_in], probs_out)
    text_in.submit(try_model, [model_box, text_in], probs_out)
    load_btn.click(load_existing, model_box, [current_id, model_info, endpoint_md])


# --------------------------------------------------------------------------- #
# API REST (FastAPI)
# --------------------------------------------------------------------------- #
api = FastAPI(title="classr API", description="Servicio de clasificación de texto para los modelos entrenados en classr.")
api.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class PredictRequest(BaseModel):
    text: str | None = None
    texts: list[str] | None = None


def _get(model_id: str) -> Pipeline:
    try:
        return load_model(model_id)
    except KeyError:
        raise HTTPException(404, f"Modelo '{model_id}' no encontrado")


@api.post("/api/models/{model_id}/predict")
def api_predict(model_id: str, body: PredictRequest):
    pipe = _get(model_id)
    if body.texts is not None:
        if len(body.texts) > MAX_BATCH:
            raise HTTPException(413, f"Máximo {MAX_BATCH} textos por petición")
        return {"model_id": model_id, "predictions": predict_texts(pipe, body.texts)}
    if body.text is None:
        raise HTTPException(422, 'Envía {"text": "..."} o {"texts": ["...", ...]}')
    return {"model_id": model_id, **predict_texts(pipe, [body.text])[0]}


@api.get("/api/models/{model_id}/predict")
def api_predict_get(model_id: str, text: str):
    return {"model_id": model_id, **predict_texts(_get(model_id), [text])[0]}


@api.get("/api/models/{model_id}")
def api_info(model_id: str):
    try:
        return load_meta(model_id)
    except KeyError:
        raise HTTPException(404, f"Modelo '{model_id}' no encontrado")


@api.get("/api/health")
def health():
    return {"status": "ok"}


app = gr.mount_gradio_app(api, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host=os.environ.get("HOST", "0.0.0.0"), port=int(os.environ.get("PORT", 7860)))
