# Parcial NLP y LLMs Avanzado – EAFIT 2026-1

Aplicación desarrollada en Streamlit para resolver las Partes 02, 03 y 04 del parcial:

- `■ Laboratorio de Parámetros`
- `■ Métricas de Similitud`
- `■ Agente Especializado`

La aplicación fue implementada sobre Gemini Developer API y combina experimentación con hiperparámetros de generación, evaluación automática de respuestas con métricas clásicas de NLP y un agente conversacional con métricas de producción en tiempo real.

## 1. Objetivo general de la aplicación

El proyecto busca mostrar, en una sola interfaz, tres capas complementarias del trabajo con LLMs:

1. Cómo cambian las respuestas cuando se modifican parámetros de generación.
2. Cómo evaluar cuantitativamente una salida generada frente a una referencia.
3. Cómo instrumentar un agente conversacional con memoria, costos, latencia y autoevaluación.

La idea es construir una pequeña plataforma experimental en la que se puedan observar, medir y documentar comportamientos del sistema.

## 2. Stack tecnológico

La solución usa las siguientes librerías principales:

- `streamlit` para la interfaz.
- `google-genai` para el acceso a Gemini.
- `plotly` para visualizaciones interactivas.
- `pandas` para estructurar resultados.
- `sentence-transformers` + `scikit-learn` para similitud coseno.
- `nltk` para BLEU.
- `rouge-score` para ROUGE-L.
- `bert-score` para BERTScore.
- `torch` como dependencia de modelos semánticos.

## 3. Estructura funcional del proyecto

El archivo principal es `app.py`. La aplicación está organizada alrededor de:

- utilidades de generación y parsing con Gemini,
- utilidades de métricas y costos,
- navegación principal por secciones,
- lógica específica de cada pestaña,
- almacenamiento de estado con `st.session_state`.

## 4. Configuración y ejecución

### Requisitos

Instalar las dependencias del archivo `requirements.txt`.

### API Key

La aplicación espera una clave `GEMINI_API_KEY`. Puede cargarse de dos maneras:

- desde `st.secrets`,
- o manualmente desde la barra lateral usando el campo tipo password.

## 5. Arquitectura técnica general

Antes de entrar pestaña por pestaña, conviene resumir algunos componentes reutilizados en toda la app:

### 5.1 Cliente de Gemini

La función `get_client(api_key)` crea el cliente de Gemini de manera centralizada. Esto evita duplicar lógica de inicialización en cada sección.

### 5.2 Configuración de generación

La función `build_generation_config(...)` construye el objeto `GenerateContentConfig` para Gemini con:

- `temperature`
- `top_p`
- `top_k`
- `max_output_tokens`
- `frequency_penalty`
- `presence_penalty`
- `system_instruction`
- `response_json_schema`
- `thinking_budget`

Esto hace que la app sea coherente: la misma forma de configurar generación se reutiliza tanto en el laboratorio, como en el evaluador automático y en el agente conversacional.

### 5.3 Extracción de texto y parsing JSON

Los LLMs no siempre devuelven estructuras limpias. Por eso se implementaron:

- `get_response_text(response)` para extraer texto del response incluso si la respuesta no viene en el campo más simple.
- `parse_json_payload(response)` para robustecer el parsing del juez estructurado.

Esta parte es clave porque la app depende de salidas JSON para el componente `LLM-as-Judge`.

### 5.4 Uso de tokens y costos

La función `get_usage_stats(response)` extrae:

- `prompt_tokens`
- `completion_tokens`
- `thought_tokens`
- `total_tokens`

Luego `estimate_cost(model_name, usage_stats)` convierte esos tokens en costo estimado con base en el pricing configurado para cada modelo Gemini.

### 5.5 Navegación de la aplicación

La navegación principal usa tres secciones:

- `SECTION_LAB`
- `SECTION_METRICS`
- `SECTION_AGENT`

Esto permite una interfaz más controlada que `st.tabs`, especialmente porque el panel lateral necesita reaccionar a la sección activa.

## 6. Pestaña 1 – Laboratorio de Parámetros

### 6.1 Objetivo

Esta pestaña permite experimentar con hiperparámetros de generación para observar cómo cambian:

- la creatividad,
- la diversidad,
- la longitud,
- la repetición,
- y la estabilidad de la respuesta.

Además incluye un experimento comparativo obligatorio con cuatro configuraciones contrastantes.

### 6.2 Implementación técnica

La pestaña tiene dos bloques principales:

1. Generación individual con prompt personalizado.
2. Experimento comparativo de 4 configuraciones.

#### Generación individual

El usuario define un prompt y ajusta manualmente:

- `temperature`
- `top_p`
- `top_k`
- `max_tokens`
- `frequency_penalty`
- `presence_penalty`

Al presionar el botón, la app llama a `generate_text(...)`, que a su vez usa Gemini con la configuración construida por `build_generation_config(...)`.

La respuesta se muestra junto con:

- tokens de entrada,
- tokens de salida,
- thinking tokens.

#### Verificación de soporte de `top_k`

Como no todos los modelos exponen las mismas capacidades, se implementó `fetch_model_capabilities(...)`, que consulta metadata del modelo para verificar si `top_k` está soportado. Si no lo está:

- el control sigue siendo visible,
- pero aparece deshabilitado,
- y la interfaz informa el motivo.

#### Experimento comparativo

La app ejecuta cuatro configuraciones fijas requeridas por la rúbrica:

- `T=0.1 / p=0.9`
- `T=1.5 / p=0.9`
- `T=0.1 / p=0.3`
- `T=1.5 / p=0.3`

Las demás variables (`top_k`, `max_tokens`, `frequency_penalty`, `presence_penalty`) se mantienen constantes según los valores actuales del panel.

Cada resultado guarda:

- etiqueta de configuración,
- texto generado,
- `n_tokens` usando `response.usage.completion_tokens`,
- `ttr`,
- latencia.

La visualización incluye:

- 4 columnas paralelas con cada salida,
- un gráfico de barras para longitud en tokens,
- un gráfico de barras para diversidad léxica,
- un campo libre de observaciones del estudiante.

### 6.3 Análisis de parámetros de esta pestaña

#### Temperatura

Controla cuánta aleatoriedad se permite en el muestreo. Conceptualmente:

- valores bajos producen respuestas más deterministas y conservadoras,
- valores altos favorecen variación y creatividad.

En el laboratorio esto es útil para mostrar el tradeoff entre precisión y originalidad.

#### Top-p

Limita la masa acumulada de probabilidad considerada en cada paso. Su efecto esperado:

- `top_p` alto deja pasar más diversidad,
- `top_p` bajo estrecha el conjunto de candidatos plausibles.

Es una forma de recorte probabilístico dinámico.

#### Top-k

Restringe el número máximo de tokens candidatos considerados en cada paso. Puede usarse junto con `top_p`, aunque para fines pedagógicos suele ser más limpio variar uno a la vez.

En esta app:

- se usa en generación individual,
- y se mantiene constante durante el experimento comparativo.

#### Max tokens

Controla el techo de longitud de la salida. Su análisis es importante porque:

- respuestas muy cortas pueden parecer incompletas,
- respuestas muy largas aumentan latencia y costo,
- también afectan la interpretación de TTR.

#### Frequency penalty

Penaliza repeticiones frecuentes. Sirve para reducir loops o redundancia.

#### Presence penalty

Penaliza la reaparición de contenido ya introducido, incentivando exploración de temas nuevos dentro de la misma respuesta.

### 6.4 Métricas analizadas en esta pestaña

#### Tokens de salida (API)

Se usa el conteo real de `completion_tokens` reportado por Gemini. Esto hace que la longitud reportada corresponda al consumo real del modelo y no a una aproximación basada en palabras.

#### TTR (Type-Token Ratio)

Se calcula como:

`tokens únicos / tokens totales`

Esta métrica aproxima diversidad léxica. No mide calidad, pero sí variedad superficial del vocabulario.

### 6.5 Interpretación académica

Esta pestaña demuestra que “más creatividad” no es automáticamente “mejor respuesta”. La utilidad real de cada configuración depende de:

- el tipo de tarea,
- el nivel de precisión esperado,
- el costo aceptable,
- y la necesidad de variedad frente a consistencia.

## 7. Pestaña 2 – Métricas de Similitud y Evaluación Automática

### 7.1 Objetivo

Esta sección compara una respuesta generada por el LLM contra un texto de referencia usando métricas automáticas y una evaluación estructurada tipo `LLM-as-Judge`.

El propósito es combinar:

- métricas de similitud superficial,
- métricas semánticas,
- y juicio cualitativo automatizado.

### 7.2 Flujo implementado

El flujo es:

1. El usuario ingresa un texto de referencia.
2. El usuario define el prompt que se enviará al modelo.
3. La app genera una respuesta candidata.
4. Calcula métricas automáticas.
5. Ejecuta un segundo llamado al modelo como juez.
6. Muestra resultados en métricas, detalle del judge y radar chart.

### 7.3 Implementación de métricas

#### Similitud coseno

Se usa `SentenceTransformer("all-MiniLM-L6-v2")`, cargado con `@st.cache_resource`. Luego:

- se generan embeddings para referencia y respuesta,
- se aplica `cosine_similarity`,
- se obtiene una similitud semántica global.

Esta es una métrica útil para medir cercanía conceptual aunque no exista coincidencia literal exacta.

#### BLEU

Se usa `nltk.translate.bleu_score`. Antes del cálculo:

- referencia e hipótesis se tokenizan con una expresión regular local,
- se aplica smoothing para evitar que secuencias cortas colapsen a cero.

BLEU es útil para capturar coincidencia de n-gramas, pero no sustituye una evaluación semántica.

#### ROUGE-L

Se implementa con `rouge-score` y captura la subsecuencia común más larga, lo cual da una idea de alineación estructural.

#### BERTScore

Se calcula con la librería `bert-score`. La interfaz permite escoger idioma:

- `es`
- `en`

Esto evita que la evaluación semántica quede rígidamente atada a un solo idioma.

#### LLM-as-Judge

Se implementa con un segundo llamado a Gemini que devuelve JSON estructurado con:

- `score`
- `veracidad`
- `coherencia`
- `relevancia`
- `fortalezas`
- `debilidades`

El parsing se hace de forma robusta para tolerar pequeñas desviaciones del formato.

### 7.4 Radar chart

La app almacena los puntajes en su escala natural:

- Cosine, BLEU, ROUGE-L, BERTScore en rango aproximado `[0,1]`
- `LLM-Judge` en rango `1-10`

La normalización ocurre solo dentro de `build_radar_chart(...)`, donde:

- `LLM-Judge` se divide por 10,
- las demás métricas se acotan a `[0,1]`.

Esto simplifica el modelo mental del código: los valores se guardan “en crudo” y solo se transforman al momento de visualizarlos.

### 7.5 Análisis de parámetros de esta pestaña

Aunque aquí no hay un “panel de parámetros” como en la pestaña 1, sí existen decisiones de configuración relevantes:

#### Parámetros de generación de la respuesta candidata

La respuesta candidata se genera con valores relativamente conservadores:

- `temperature = 0.3`
- `top_p = 0.9`
- `top_k = 40`
- `max_output_tokens` amplio

La razón de esta configuración es metodológica: si se quiere comparar una salida contra una referencia, conviene reducir ruido creativo excesivo y priorizar estabilidad semántica.

#### Idioma de BERTScore

Es un parámetro crítico porque la comparación semántica depende del idioma dominante del texto.

#### Configuración del juez

El judge se ejecuta con:

- temperatura muy baja,
- salida JSON estructurada,
- y presupuesto de thinking controlado.

La idea es maximizar consistencia y minimizar respuestas narrativas no estructuradas.

### 7.6 Interpretación académica

Esta pestaña muestra que evaluar LLMs no depende de una sola métrica:

- BLEU puede ser bajo aunque haya buena equivalencia semántica.
- Cosine puede ser alto incluso con redacción distinta.
- ROUGE-L favorece alineación estructural.
- BERTScore es más sensible a similitud semántica token a token.
- LLM-as-Judge agrega una capa cualitativa cercana a evaluación humana automatizada.

En conjunto, esto permite una evaluación más rica que el simple “me gustó / no me gustó”.

## 8. Pestaña 3 – Agente Especializado

### 8.1 Objetivo

Construir un agente conversacional especializado en ML con:

- identidad definida,
- memoria conversacional,
- parámetros configurables,
- métricas de producción por turno,
- historial acumulado de desempeño.

### 8.2 Identidad del agente

El agente está definido por `SYSTEM_PROMPT`, que establece:

- nombre del agente (`MLBot`),
- dominio temático,
- estilo de explicación,
- límites de alcance,
- comportamiento frente a preguntas fuera del dominio.

Esto es importante porque una buena aplicación con LLM no solo genera texto: también restringe y encuadra el rol del sistema.

### 8.3 Memoria conversacional

La memoria se maneja con `st.session_state["agent_history"]`.

Cada turno se guarda como una lista de mensajes con:

- `role = user`
- `role = assistant`

Luego `history_to_gemini_contents(...)` transforma ese historial al formato esperado por Gemini para reconstruir el contexto conversacional.

### 8.4 Parámetros del agente

Los controles del agente aparecen en la barra lateral solo cuando la sección del agente está activa:

- `Temperatura agente`
- `Max tokens agente`

Esto evita duplicación visual innecesaria y mantiene la barra lateral contextual.

### 8.5 Flujo de un turno conversacional

Cuando el usuario escribe una pregunta:

1. Se crea la configuración del chat con `build_generation_config(...)`.
2. Se reconstruye la conversación previa.
3. Se envía el nuevo mensaje.
4. Se mide la latencia real.
5. Se extraen tokens y costo.
6. Se ejecuta un `LLM-as-Judge` para la respuesta del agente.
7. Se guarda todo en el historial.
8. Se actualiza la visualización.

### 8.6 Métricas de producción implementadas

Por turno se almacenan:

- `Latencia (s)`
- `TPS`
- `Tokens entrada`
- `Tokens salida`
- `Thinking tokens`
- `Costo respuesta USD`
- `Costo judge USD`
- `Costo total USD`
- `LLM-Judge`
- sub-scores del judge

Esto convierte la pestaña del agente en una pequeña consola de observabilidad.

### 8.7 Históricos y visualización

El historial se convierte en `DataFrame` y se visualiza con Plotly en pestañas:

- Latencia
- TPS
- LLM-Judge
- Costo

Además se calcula `Costo acumulado USD` para ver crecimiento del gasto por sesión.

### 8.8 Análisis de parámetros de esta pestaña

#### Temperatura agente

Modula el estilo de respuesta del tutor:

- baja temperatura: respuestas más estables y técnicas,
- alta temperatura: respuestas más variadas, potencialmente más pedagógicas, pero menos consistentes.

Para un tutor académico, valores medios o bajos suelen ser más apropiados.

#### Max tokens agente

Controla profundidad y extensión.

Impacto esperado:

- más tokens: respuestas más completas y explicativas,
- menos tokens: respuestas más rápidas y baratas, pero posiblemente incompletas.

#### Modelo Gemini seleccionado

El modelo elegido impacta directamente:

- calidad percibida,
- latencia,
- costo,
- estabilidad del judge.

#### Judge del agente

El agente se autoevalúa con un judge más estricto y una calibración adicional. Esto busca evitar que todas las respuestas obtengan `10/10` por complacencia del evaluador.

### 8.9 Interpretación académica

Esta pestaña muestra una verdad importante de producción:

no basta con que el agente “responda bien”. También importa medir:

- cuánto tarda,
- cuántos tokens consume,
- cuánto cuesta,
- cómo evoluciona turno a turno,
- y cómo se autoevalúa bajo un criterio explícito.

## 9. Decisiones de diseño importantes

### 9.1 Thinking budget controlado

Se usa `thinking_budget=0` para mantener más previsibles:

- latencia,
- costo,
- longitud de salida,
- consistencia de medición.

### 9.2 Uso de puntajes crudos

En la pestaña de métricas, los scores se almacenan en su escala natural y solo se normalizan al construir el radar. Esto mejora legibilidad del flujo interno.

### 9.3 Robustez ante errores de JSON

El judge estructurado no depende de una única estrategia de parsing. Esto es valioso porque los LLMs, incluso con schema, pueden introducir ruido de formato.

### 9.4 Sidebar contextual

La barra lateral no muestra siempre los mismos controles: reacciona a la sección activa. Esto mejora claridad de UX.

## 10. Posibles mejoras futuras

Aunque la solución cumple muy bien el alcance del parcial, existen varias extensiones posibles:

- agregar capturas de pantalla al README,
- exportar resultados del laboratorio a CSV,
- guardar histórico de experimentos con timestamp,
- incorporar más modelos Gemini con pricing diferenciado,
- añadir comparación entre judge humano y judge automático,
- permitir descargar el historial del agente.

## 11. Conclusión

La aplicación implementa un flujo completo de trabajo con LLMs:

- generación controlada,
- medición cuantitativa,
- evaluación semántica,
- juicio automatizado,
- y monitoreo de un agente especializado.


