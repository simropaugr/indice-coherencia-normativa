# Pipeline de Análisis Normativo Automatizado

Este repositorio contiene un flujo de trabajo (pipeline) automatizado para la obtención de un índice de coherencia de cualquier texto con un paradigma ideológico o sistema de valores descrito con anterioridad. Utiliza Modelos de Lenguaje Grande (LLMs) y procesamiento de lenguaje natural (Embeddings).

# Autoría y Financiación

Todas las fases que se muestran en este repositorio han sido diseñadas y aplicadas por primera vez en el trabajo realizado por **Manuel Freire Ferrando**, y tutorizado por **Rosa María Soriano Miras** y **Harlan Koff** en el marco del Proyecto **"¿Y después del consumo masivo de las prendas de vestir? Economía circular en la industria de la moda a través de la Ciencia Ciudadana” (PID2023-148791NB-I00) financiado por MICIU/ AEI /10.13039/501100011033  y por FEDER, UE**

![Proyecto PID2023-148791NB-I00 financiado por MICIU/AEI/10.13039/501100011033 y por FEDER, UE](docs/financiacion_AEI.png)

## Requisitos Previos

1. **Python 3.8+** instalado en el sistema.
2. Una clave de API válida de **OpenAI** (con saldo disponible).

## Instalación

1. Clona o descarga este repositorio en tu ordenador.
2. Abre una terminal en la carpeta raíz del proyecto y ejecuta el siguiente comando para instalar las dependencias necesarias:
   ```bash
   pip install -r requirements.txt
   ```

## Guía de uso paso a paso

### PASO 0: Preparación de documentos
Comprueba que la carpeta `data` contiene las carpetas `data/01_raw/` y `data/03_manual/`. Si no existen, créalas. 
Antes de ejecutar ningún código, deposita en la ruta `data/01_raw/`:
1. Los documentos que se pretendan analizar en formato `.pdf`
2. Una lista de actividades definitorias de la dimensión sobre la que se quiere medir el nivel de coherencia en formato .md (ejemplo de nombre: `actividades_economia.md`)
3. Un prompt de instrucciones para cada dimensión. Debe indicar las instrucciones para el cálculo de la dirección por parte del LLM en formato `.txt`. Se aconseja mantener la siguiente estructura básica:

    ```text
    """
    Eres un sistema experto en...
    Tu tarea es analizar la relación lógica y de dirección entre dos textos: un "Párrafo" (extenso y administrativo) y una "Directriz" (concisa y clara).

    Debes clasificar la relación en una de estas tres categorías numéricas:
    1 = Misma dirección (El párrafo promueve, facilita o comparte el principio subyacente de la directriz).
    -1 = Dirección opuesta (El párrafo obstaculiza, contradice o va en contra del objetivo de la directriz).
    0 = Neutro (No hay relación, el párrafo es puramente descriptivo/definitivo, o los contextos/sujetos no coinciden).

    Metodología de análisis obligatoria (Procesamiento en orden estricto):
    Para determinar el resultado, debes evaluar los textos utilizando la Directriz como "Ancla Lógica" siguiendo estos pasos secuenciales:
    1. Aislar el núcleo de la directriz: Extrae el objetivo central, la acción requerida y los sujetos u objetos económicos/medioambientales afectados.
    2. Desglosar el párrafo: Analiza las acciones o consecuencias reales que promueve el texto largo, filtrando la retórica administrativa.
    3. Evaluación de Impacto: Responde estrictamente a la pregunta: "Si lo que dice el párrafo se lleva a cabo, ¿facilita, obstaculiza o es indiferente para el cumplimiento de la directriz?".

    Reglas críticas de análisis:
    1. Prioridad del principio subyacente: No te dejes llevar por oposiciones verbales superficiales. Si un texto prohíbe algo y el otro regula su implantación bajo criterios de proporcionalidad o equidad, apuntan en la misma dirección (1).
    2. Regla del único salto lógico: Se permiten relaciones indirectas de primer grado solo si el Párrafo menciona un elemento técnico, operativo o económico indispensable para que ocurra la Directriz. Si la conexión requiere más de un supuesto intermedio, clasifica como Neutro (0).
    3. Coincidencia estricta de alcance: Si el Párrafo y la Directriz expresan acciones similares pero aplicadas a sujetos, objetos, recursos o ámbitos regulatorios distintos, la relación es Neutra (0).
    4. Casos mixtos: En caso de que un párrafo contenga ambigüedad o dos direcciones regulatorias, prioriza la primera dirección clara que aparezca en el texto.

    Formato de salida estricto (JSON):
    Devuelve única y exclusivamente un objeto JSON con dos claves. En el campo "razonamiento" debes reflejar de forma muy sintética los pasos de la metodología (máximo 25 palabras):
    "razonamiento": Breve justificación que conecte el núcleo de la directriz con el impacto real del párrafo.
    "resultado": El número entero (1, -1 o 0).

    Ejemplos de referencia (Few-Shot):

    Ejemplo 1 (Relación indirecta de primer grado - Misma dirección):...

    Ejemplo 2 (Oposición superficial / Principio compartido):...

    Ejemplo 3 (Desajuste de alcance / Objetos diferentes):...

    Ejemplo 4 (Dirección contraria indirecta):...
    """
    ```

### Paso 1: Procesamiento y Embeddings
Abre y ejecuta el cuaderno `notebooks/01_preparacion_datos.ipynb`.
* El script extraerá el texto de los PDFs alojados en la carpeta de entrada y los convertirá a formato Markdown. IMPORTANTE: Esta conversión mantiene los párrafos intactos excepto los separados por saltos de página. Fusionar los párrafos divididos por saltos de página requiere una acción externa.
* Seleccionará el documento de muestra que le indiques (un texto representativo o aleatorio) y extraerá sus párrafos para generar un archivo CSV con las columnas ID_Parrafo y Texto (en la que cada fila es un párrafo del texto original).
* Paralelamente, calculará los vectores numéricos (embeddings) de cada lista de actividades definitorias usando la API de OpenAI.

### Paso 2: Clasificación Manual
* Ve a la carpeta `data/02_interim/` y abre el archivo CSV generado en el paso anterior. Verás que ya contiene los párrafos del documento de muestra.
* En las columnas correspondientes a las dimensiones, etiqueta manualmente cada párrafo (escribiendo un 1 si posee relación con la dimensión, o un 0 en caso contrario).
* Guarda este archivo con tus evaluaciones completadas en la carpeta `data/03_manual/`.

### Paso 3: Cálculo de Umbral Óptimo
Abre y ejecuta el cuaderno `notebooks/02_calculo_umbral.ipynb`
* El script calculará la similitud del coseno **exclusivamente** entre los párrafos del documento de muestra que clasificaste manualmente y las actividades de la dimensión evaluada.
* Probará distintos valores de umbral, asignando un 1 a los cosenos de similitud iguales o mayores y un 0 a los que estén por debajo del umbral. Cruzará tu etiquetado manual (1 y 0) con estos cálculos de similitud para evaluar el rendimiento y encontrar el punto de corte (umbral) óptimo. Obtendrás un resultado priorizando el equilibrio (F1-Score), y otro priorizando la detección exhaustiva (F2-Score).
* **Elige y anota el umbral óptimo devuelto por pantalla**, ya que los necesitarás para el siguiente paso.

### Paso 4: Análisis Principal y Cálculo de Índices
Abre y ejecuta el cuaderno `notebooks/03_analisis_principal.ipynb`.
* **ATENCIÓN**: Antes de ejecutar la celda de cálculo de similitudes, introduce los umbrales que anotaste en el Paso 3 dentro de la variable umbrales_aplicar.
* **ATENCIÓN:** Antes de ejecutar la celda de clasificación LLM, asegúrate de escribir el nombre exacto de tus archivos `.txt` (los que contienen el prompt) en el diccionario `archivos_prompts` para que el código asigne el contexto correcto a cada dimensión
* **ATENCIÓN**: Antes de ejecutar la celda de métricas finales, introduce el número total de actividades y el umbral al cuadrado en la variable config_indices.
* A medida que el cuaderno se ejecuta, generará secuencialmente 4 archivos en la carpeta `data/05_processed/` por cada ley y dimensión analizada:

   1. **`..._matches_original.csv`**: Archivo resultante de calcular el coseno de similitud entre cada párrafo de un texto y cada actividad de una dimensión. En él se almacenan únicamente los emparejamientos con una similitud igual o superior al umbral.
   2. **`..._matches_limpio.csv`**: Una versión depurada del archivo original. El sistema elimina automáticamente los emparejamientos en los que el párrafo del texto tenga menos de 10 palabras (títulos, fragmentos vacíos), eliminando ruido estadístico y emparejamientos sin contexto.
   3. **`..._matches_direccion.csv`**: Este archivo añade a los emparejamientos limpios dos columnas nuevas: la dirección de cada emparejamiento (1 para sinérgica, 0 para neutra, -1 para contrapuesta), y el razonamiento de la API para justificar dicha clasificación.
   4. **`..._matches_indice.csv`**: El archivo definitivo. Añade al archivo con la dirección tres nuevas columnas: 
      - Puntuacion: resultante de multiplicar el valor de la dirección por el valor de la similitud.
      - Puntuacion al cuadrado: Resultante de elevar al cuadrado el valor de Puntuacion.
      - Índice: Resultante de sumar el valor de la columna Puntuacion al cuadrado de todos los emparejamientos, y dividirlo entre el número total de actividades de la dimensión multiplicado por el valor de umbral al cuadrado. El resultado representa cuántas veces la ley supera el esfuerzo mínimo necesario para respaldar todas las actividades de una dimensión (es único para todo el documento).

## Interpretación de resultados
El valor que indica lo coherente que es un párrafo en concreto con respecto a la dimensión con la que se está trabajando es el valor de la columna "Puntuación". Obtenemos así una escala final de valores teóricos entre el -1 y el 1 en el que los valores negativos indican incoherencia y los positivos representan coherencia para con la dimensión que se esté midiendo. Cuanto más alejados del 0 estén esos valores, mayor es la intensidad con la que ese párrafo ejerce su (in)coherencia con la dimensión. A nivel práctico, el valor 
mínimo de intensidad que podrá tomar cada uno de nuestros párrafos será el valor que hemos establecido como umbral.

El valor que indica lo coherente que es un texto al completo con respecto a la dimensión con la que se está trabajando es el valor de la columna "Índice". La fórmula diseñada para obtenerlo unifica el valor del índice de coherencia entre cualquier dimensión, incluida cualquier dimensión futura que se quiera medir por diferente que sea su campo léxico o el número de actividades y el umbral que precise. 
Aunque a nivel teórico la fórmula devuelve un valor que oscila de -∞ a ∞, las características de nuestros datos nos permiten fijar una escala cualitativa para favorecer su interpretabilidad. Un resultado de 1 significaría que la ley dedica a la dimensión una fuerza equivalente a tener un párrafo coherente por cada una de las actividades definitorias de la dimensión (y viceversa para el -1). Teniendo en cuenta que el 0 muestra irrelevancia con respecto a la dimensión que se está midiendo, que los resultados positivos representan la 
coherencia y los negativos la incoherencia, y que un resultado próximo a 1 debe implicar un nivel de coherencia relativa (incoherencia relativa para -1), se propone la siguiente escala interpretativa: 

| Rango de resultados | Interpretación |
| :---: | :--- |
| I ≥ 2,5 | Coherencia muy alta |
| 1 ≤ I < 2,5 | Coherencia sólida |
| 0,25 ≤ I < 1 | Coherencia relativa |
| -0,25 ≤ I < 0,25 | Irrelevancia |
| -1 ≤ I < -0,25 | Incoherencia relativa |
| I ≤  -2,5 | Incoherencia muy alta |

## Licencia

Este código se distribuye bajo la licencia MIT (ver el archivo `LICENSE`). Puede usarse, modificarse y redistribuirse libremente, siempre que se mantenga el aviso de autoría.

## Cómo citar

Si utilizas este código, cítalo de la siguiente forma (también disponible en el botón *Cite this repository* de GitHub):

Freire Ferrando, M. (2026). *Pipeline de Análisis Normativo Automatizado* (Versión 1.0.0) [Software]. Proyecto SIMRopa, Universidad de Granada. https://github.com/simropaugr/indice-coherencia-normativa
