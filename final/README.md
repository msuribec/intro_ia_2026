# My-Expense Auditor

Aplicación en Streamlit para auditar gastos a partir de recibos, categorizarlos con Gemini, editar los resultados manualmente, generar recomendaciones de ahorro, analizar presupuestos mensuales y consultar el historial con búsqueda semántica tipo RAG.

## Qué hace la app

La aplicación permite:

- Cargar una API key de Google Gemini.
- Subir un archivo de categorías personalizado en `.txt` o `.csv`, o usar categorías por defecto.
- Subir fotos de recibos para extraer vendedor, fecha, moneda, ítems, total y totales por categoría.
- Editar manualmente cada recibo después de la extracción.
- Generar consejos de ahorro por recibo y convertirlos a audio con `gTTS`.
- Exportar categorías y gasto histórico a CSV.
- Importar historial de gasto desde CSV.
- Visualizar dashboards agregados con Plotly.
- Definir presupuestos mensuales por categoría y compararlos con el gasto real.
- Consultar el historial de recibos mediante un chat con recuperación semántica sobre embeddings.

## Stack tecnológico

- `Streamlit` para la interfaz y el flujo reactivo.
- `google-generativeai` para Gemini, tanto en visión como en generación de texto y embeddings.
- `pandas` para validación, agregación y transformación de datos.
- `Plotly` para visualizaciones.
- `Pillow` para abrir imágenes de recibos.
- `gTTS` para sintetizar audio de los consejos.
- `faiss-cpu` y `numpy` para el índice vectorial del buscador RAG.

## Estructura del proyecto

- `app.py`: aplicación principal de Streamlit, UI, estado, extracción, edición, dashboards, presupuestos y chat RAG.
- `rag.py`: construcción del índice vectorial, embeddings, recuperación semántica y generación de respuestas sobre el historial.
- `requirements.txt`: dependencias del proyecto.
- `categories_example.txt`: ejemplo de archivo de categorías.
- `Final_Projects_EAFIT_IA_MsC_2026_1.pdf`: documento externo incluido en la carpeta, no forma parte del flujo principal del código.

## Flujo general de la aplicación

1. El usuario ingresa su API key en la barra lateral.
2. La app exige aprobar un conjunto de categorías antes de continuar.
3. Con las categorías aprobadas, se habilitan las pestañas principales.
4. Los recibos procesados se guardan en `st.session_state["receipt_history"]`.
5. A partir de ese historial, la app habilita exportaciones, recomendaciones, dashboard, presupuesto y búsqueda semántica.

## Gestión de estado en Streamlit

La app usa `st.session_state` como fuente principal de verdad. El bootstrap de estado ocurre en `_init_state()` y crea, entre otras, estas claves:

- `categories_approved`, `categories_signature`, `approved_categories`: control del flujo de categorías.
- `receipt_history`: historial completo de recibos analizados o importados.
- `editing_receipt_index`: identifica qué recibo está en modo edición.
- `history_category_suggestions`, `history_purchase_tips`: resultados generados desde el historial completo.
- `category_budgets`, `selected_budget_month`: estado del módulo de presupuesto.
- `vector_store`, `rag_chat_history`, `rag_receipt_count`: estado del buscador RAG.
- `widget_seed`: fuerza el refresco de widgets al reiniciar la sesión.

Además, `restart_streamlit_app()` limpia `st.cache_data`, `st.cache_resource` y reinicia por completo la sesión.

## Features del Streamlit y cómo fueron implementadas

### 1. Configuración inicial y barra lateral

### API key de Gemini

En el sidebar se usa `st.text_input(..., type="password")` para capturar la API key. Mientras no exista una key, la app no habilita la carga de categorías ni el resto del flujo.

### Carga de categorías

Se usa `st.file_uploader` para aceptar archivos `.txt` o `.csv`. La función `parse_categories()`:

- Lee texto plano línea por línea si el archivo es `.txt`.
- Si es `.csv`, intenta encontrar columnas llamadas `category`, `categoria`, `categories` o `categorias`.
- Si no encuentra una columna reconocida, usa la primera columna del CSV.

La aprobación de categorías se maneja con:

- `categories_signature` para detectar si el archivo cambió.
- `categories_approved` para obligar al usuario a confirmar el set cargado.
- `approved_categories` para fijar el conjunto activo.

También existe una ruta alternativa para usar `default_categories` cuando el usuario no sube archivo propio.

### Exportación de datos

Desde el sidebar se implementan dos descargas con `st.download_button`:

- Categorías aprobadas en CSV mediante `build_categories_export_csv()`.
- Historial de gasto en CSV mediante `build_spending_export_csv()`.

Ambas funciones generan `DataFrame` de `pandas` y luego serializan con `dataframe_to_csv_bytes()`.

### Importación de historial

La app permite importar un CSV de historial con `st.file_uploader` y un botón de acción. La validación ocurre en `parse_uploaded_spending_history()`, que exige estas columnas:

- `receipt_index`
- `vendor`
- `date`
- `currency`
- `item`
- `price`
- `category`
- `receipt_total`

La función:

- Normaliza textos vacíos.
- Convierte `price` y `receipt_total` a numérico.
- Rechaza filas con precio inválido, ítem vacío o categoría vacía.
- Reagrupa por `receipt_index`.
- Reconstruye la estructura interna de cada recibo con `rebuild_receipt_data()`.

Los recibos importados quedan en `receipt_history` con `image_bytes = None` y `source = "imported_csv"`.

### 2. Extracción de recibos

La pestaña `📨 Receipts` funciona como una interfaz estilo chat:

- Cada recibo del historial se renderiza con `st.chat_message("user")` y `st.chat_message("assistant")`.
- El usuario sube nuevas imágenes con `st.file_uploader`.
- La imagen se previsualiza con `st.image`.

La extracción se implementa en `analyze_receipt()`:

- Selecciona un modelo Gemini con `pick_supported_model()`.
- Abre la imagen con `PIL.Image`.
- Construye un prompt que obliga al modelo a:
  - extraer vendedor,
  - detectar fecha,
  - detectar moneda,
  - listar ítems y precios,
  - asignar categorías,
  - calcular total y totales por categoría.
- Exige como salida un JSON puro.
- Limpia cercas Markdown con expresiones regulares antes de hacer `json.loads`.

El resultado se guarda en `receipt_history` como un diccionario con:

- `image_bytes`
- `data`
- `audio_bytes`

Dentro de `data` viven `vendor`, `date`, `currency`, `items`, `total`, `category_totals`, `savings_tip` y `tip_language`.

### 3. Visualización y edición manual de recibos

Cada recibo procesado se muestra con:

- `st.metric` para vendedor, fecha y total.
- `st.dataframe` para el detalle de ítems.
- Un gráfico de barras de gasto por categoría con Plotly.

La edición manual se hace con `render_receipt_editor()` usando:

- `st.expander`
- `st.form`
- `st.text_input` para vendedor, fecha y moneda
- `st.data_editor` para editar ítems dinámicamente

La tabla editable define columnas tipadas con `st.column_config`:

- `TextColumn` para nombre del ítem.
- `NumberColumn` para precio.
- `SelectboxColumn` para categoría.

La validación se centraliza en `normalize_edited_items()`, que:

- ignora filas totalmente vacías,
- exige nombre,
- exige categoría válida,
- exige precio numérico y no negativo,
- obliga a guardar al menos un ítem válido.

Cuando el usuario guarda, `rebuild_receipt_data()` recalcula:

- total general,
- totales por categoría,
- valores normalizados de vendedor, fecha y moneda.

Además, limpia cualquier tip anterior para evitar inconsistencias después de editar.

### 4. Consejos de ahorro por recibo y audio

Cada recibo puede generar un consejo individual con el botón `Generate tip based on receipt`.

La lógica vive en `generate_savings_tip()`:

- serializa los datos del recibo a JSON,
- le pide a Gemini un único consejo corto y accionable,
- le pide también el código de idioma.

Luego `generate_audio()`:

- valida si el idioma está soportado por `gTTS`,
- usa español como fallback,
- sintetiza el audio en memoria con `io.BytesIO`,
- devuelve los bytes del MP3 para reproducirlos con `st.audio`.

### 5. Dashboard agregado

La pestaña `📊 Dashboard` resume todo el historial cargado.

Primero calcula KPIs agregados:

- cantidad de recibos,
- gasto total acumulado,
- categoría con mayor gasto.

Después genera estas visualizaciones con Plotly:

- pie chart de participación del gasto por categoría,
- bar chart de total por recibo,
- stacked bar chart con desglose por categoría dentro de cada recibo.

La información agregada se construye recorriendo `receipt_history` y consolidando `category_totals` y `total` por recibo.

### 6. Sugerencias basadas en todo el historial

La pestaña `💡 Suggestions` usa el historial completo, no un recibo individual.

### Sugerencia de nuevas categorías

`generate_category_suggestions()`:

- resume el historial con `build_history_summary()`,
- envía a Gemini el historial y la lista actual de categorías,
- pide hasta 5 categorías nuevas,
- exige que no repita categorías existentes,
- pide una razón breve por sugerencia.

El resumen construido por `build_history_summary()` contiene:

- recibos,
- ítems,
- tiendas más frecuentes,
- totales por categoría.

### Tips de ahorro globales

`generate_history_tips()`:

- reutiliza `build_history_summary()`,
- pide 3 tips concretos y accionables,
- detecta idioma,
- retorna la lista limpia de tips.

Cada tip se convierte a audio con `generate_audio()` y luego se renderiza con `st.info` y `st.audio`.

### 7. Presupuesto mensual

La pestaña `💸 Budget` es uno de los módulos más completos de la app.

### Preparación de datos

`build_budget_source_data()` transforma el historial en dos `DataFrame`:

- `items_df`: una fila por ítem comprado.
- `receipts_df`: una fila por recibo.

Para eso usa `parse_receipt_date()`, que intenta parsear fechas mixtas y tolera formatos ambiguos, vacíos o inválidos. Los recibos sin fecha usable se excluyen del análisis y se contabilizan en `skipped_receipts`.

### Selección de mes y presupuesto por categoría

La app:

- calcula un mes por defecto con `get_default_budget_month()`,
- garantiza un valor válido con `ensure_selected_budget_month()`,
- deja elegir año y mes con `st.selectbox`,
- guarda un presupuesto compartido por categoría en `st.session_state["category_budgets"]`.

Los montos se editan con un `st.form` y múltiples `st.number_input`.

`sync_category_budgets()` asegura que el diccionario de presupuestos siempre esté sincronizado con las categorías aprobadas.

### Analítica mensual

`build_month_budget_analytics()` calcula:

- gasto real por categoría en el mes seleccionado,
- presupuesto por categoría,
- varianza entre presupuesto y gasto real,
- gasto diario,
- curva ideal acumulada,
- curva real acumulada,
- moneda activa del periodo.

### Visualizaciones de presupuesto

Con esa analítica, la pestaña muestra:

- métricas de presupuesto total, gasto real y porcentaje usado,
- gráfico de barras agrupadas `Budget vs Actual`,
- gráfico de barras apiladas por escenario,
- línea acumulada `Ideal Cumulative vs Actual Cumulative`,
- pie chart del gasto real del mes,
- gráfico de varianza por categoría,
- gauge de consumo total frente al presupuesto,
- heatmap calendario del gasto diario.

El heatmap se genera en `build_budget_heatmap_figure()` con `plotly.graph_objects.Heatmap` y una matriz construida con `calendar.monthdayscalendar`.

### 8. Búsqueda semántica de recibos con RAG

La pestaña `🔍 Receipt Search` implementa un chat sobre el historial de recibos.

### Indexación

Cuando cambia la cantidad de recibos, la app reconstruye el índice de manera lazy:

- crea una instancia de `ReceiptVectorStore`,
- recorre `receipt_history`,
- convierte cada recibo a texto con `receipt_to_text()`,
- genera embeddings con `genai.embed_content`,
- normaliza vectores con `faiss.normalize_L2`,
- los guarda en un índice `faiss.IndexFlatIP`.

La dimensión de embedding está fijada en `768` para que coincida entre Gemini y FAISS.

### Recuperación

`ReceiptVectorStore.search()`:

- embebe la pregunta del usuario como `retrieval_query`,
- hace búsqueda top-k sobre FAISS,
- devuelve los textos más relevantes.

También existe `_search_receipts()` para recuperar los diccionarios originales y no solo su representación textual.

### Generación de respuesta

`answer_question()` implementa el paso generativo:

- recupera los recibos más relevantes,
- construye un bloque de contexto,
- obliga al modelo a responder solo con la información recuperada.

Además aplica una estrategia híbrida para preguntas numéricas. Si detecta palabras como `total`, `sum`, `how much`, `spent`, `cost`, `amount` o `price`:

- recupera los recibos más relevantes,
- suma en Python los `total` de esos recibos,
- inserta ese valor como ancla adicional en el prompt.

Esto reduce el riesgo de respuestas numéricas inconsistentes.

### Interfaz de chat

La UI usa:

- `st.chat_message` para el historial de conversación,
- `st.chat_input` para la pregunta,
- un botón para limpiar el chat con `st.button`.

El historial del chat vive en `st.session_state["rag_chat_history"]`.

### 9. Modelos y estrategia de compatibilidad con Gemini

La función `pick_supported_model()` consulta `genai.list_models()` y busca, en orden, estos modelos:

- `models/gemini-2.5-flash`
- `models/gemini-2.5-flash-lite`
- `models/gemini-2.0-flash`
- `models/gemini-1.5-flash`

Si no encuentra ninguno de los preferidos, usa cualquier modelo Gemini que soporte `generateContent` y no sea de visión explícita. Si la lista falla, el código hace fallback a `models/gemini-2.0-flash`.

### 10. Validaciones y manejo de errores

La app incluye varias defensas importantes:

- validación de columnas y tipos en el import de CSV,
- limpieza de respuestas JSON de Gemini para soportar cercas Markdown,
- `try/except` alrededor de extracción, tips y sugerencias,
- mensajes de error con `st.error`,
- interrupción del flujo con `st.stop()` cuando faltan prerrequisitos,
- advertencias cuando algunos recibos no pueden indexarse o no tienen fecha parseable.

## Esquema interno de un recibo

Cada elemento de `receipt_history` sigue esta estructura general:

```python
{
    "image_bytes": bytes | None,
    "source": "imported_csv",  # solo en recibos importados
    "audio_bytes": bytes | None,
    "data": {
        "vendor": "string",
        "date": "string",
        "currency": "string",
        "items": [
            {"name": "string", "price": 0.0, "category": "string"}
        ],
        "total": 0.0,
        "category_totals": {"Category": 0.0},
        "savings_tip": "string",
        "tip_language": "es"
    }
}
```

## Cómo ejecutar el proyecto

1. Instalar dependencias:

```bash
pip install -r requirements.txt
```

2. Lanzar la aplicación:

```bash
streamlit run app.py
```

3. En la interfaz:

- ingresar una API key de Google Gemini,
- cargar o aprobar categorías,
- subir recibos o importar historial.

## Dependencias declaradas

El archivo `requirements.txt` incluye:

- `streamlit>=1.32.0`
- `google-generativeai>=0.7.0`
- `gtts>=2.5.0`
- `pandas>=2.0.0`
- `Pillow>=10.0.0`
- `plotly>=5.0.0`
- `faiss-cpu>=1.7.4`
- `numpy>=1.24.0`

## Observaciones finales

El proyecto está implementado como una app monolítica en `app.py`, con `rag.py` como módulo especializado para recuperación semántica. Para el tamaño actual funciona bien, pero si creciera sería natural separar:

- componentes de UI,
- servicios de Gemini,
- validadores y parsers,
- lógica analítica,
- capa de persistencia.

Tal como está hoy, la aplicación ya cubre un flujo completo de captura, auditoría, análisis y consulta inteligente del gasto personal.
