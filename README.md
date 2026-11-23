# classr

App web (Gradio + FastAPI) para clases de ML en texto. El estudiante:

1. **Sube un dataset** (CSV, TSV, JSON, JSONL o XLSX) con una columna de texto y una de etiqueta. Las columnas se detectan solas y se pueden corregir.
2. **Entrena** un modelo ligero: TF-IDF (palabras 1-2 gramas o caracteres 2-5 gramas) + regresión logística o Naive Bayes. Ve accuracy, F1 macro, métricas por clase, matriz de confusión y los términos más característicos de cada clase.
3. **Lo prueba** escribiendo textos en una caja.
4. **Obtiene una URL** de servicio web para usarlo desde n8n, curl, Python, etc.

Cada modelo recibe un ID aleatorio (p. ej. `chhYqnef`), así los estudiantes no se pisan. Los modelos se guardan en `models/` y sobreviven reinicios.

## Ejecutar

```powershell
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
.venv\Scripts\python app.py
```

Abre <http://localhost:7860>. Hay un dataset de prueba en `ejemplos/sentimiento.csv`.

Variables de entorno: `PORT` (7860), `HOST` (0.0.0.0), `MODELS_DIR` (./models), `MAX_ROWS` (200000), `CACHE_SIZE` (32: cuántos modelos se mantienen cargados en memoria; los demás se leen desde disco al pedirlos y se descarta el usado hace más tiempo) y `PUBLIC_URL` (URL pública que se muestra a los estudiantes; si no se define se deduce de la petición, incluidos los headers `X-Forwarded-*` de un proxy).

## API

| Método | Ruta | Cuerpo / parámetros |
|---|---|---|
| POST | `/api/models/{id}/predict` | `{"text": "..."}` o `{"texts": ["...", "..."]}` (máx. 1000) |
| GET | `/api/models/{id}/predict?text=...` | — |
| GET | `/api/models/{id}` | info y métricas del modelo |
| GET | `/docs` | Swagger |

Respuesta:

```json
{"model_id": "chhYqnef", "label": "positivo", "confidence": 0.79,
 "probabilities": {"negativo": 0.13, "neutro": 0.08, "positivo": 0.79}}
```

### n8n

Nodo **HTTP Request** → Method `POST`, URL `https://<servidor>/api/models/<id>/predict`, Body JSON `{{ { "text": $json.text } }}`. La etiqueta queda en `{{ $json.label }}`.

> n8n tiene que poder alcanzar el servidor: si n8n corre en la nube, `localhost` no sirve. Despliega en un servidor con IP/dominio público o abre un túnel (`cloudflared tunnel --url http://localhost:7860`) y define `PUBLIC_URL` con esa dirección.

> En PowerShell / Git Bash de Windows, `curl -d` con tildes puede enviar el texto mal codificado; usa `--data-binary @archivo.json` con un archivo UTF-8 o prueba desde n8n/Python.

## Docker

Con Docker Compose (recomendado):

```powershell
docker compose up -d --build
```

O a mano:

```powershell
docker build -t classr .
docker run -d --name classr -p 7860:7860 -v classr-models:/data/models -e PUBLIC_URL=https://mi.dominio classr
```

- Los modelos se guardan en el volumen `classr-models` (`/data/models` dentro del contenedor), así que sobreviven a reinicios y a reconstrucciones de la imagen.
- Para fijar la URL que ven los estudiantes con Compose: `$env:PUBLIC_URL="https://mi.dominio"; docker compose up -d`, o un archivo `.env` con `PUBLIC_URL=...`.
- El contenedor corre como usuario sin privilegios y tiene un healthcheck en `/api/health`.
- Detrás de un proxy (nginx, Caddy, Traefik) basta con reenviar los headers `X-Forwarded-Host` y `X-Forwarded-Proto` para que la URL mostrada sea la correcta sin definir `PUBLIC_URL`.

## Notas

- Las métricas se calculan con un split estratificado (`random_state=42`); el modelo publicado se reentrena con el 100 % de los datos.
- No hay autenticación: cualquiera que conozca el ID puede usar el modelo. Para un curso está bien; no lo expongas con datos sensibles.
