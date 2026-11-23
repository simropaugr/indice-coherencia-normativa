import json
import pandas as pd
from pathlib import Path
from openai import OpenAI
from typing import Tuple, Union, Optional

def analizar_textos_llm(
    client: OpenAI, 
    modelo: str, 
    system_prompt: str, 
    parrafo: str, 
    directriz: str
) -> Tuple[Optional[str], Optional[int]]:
    """
    Llama a la API de OpenAI, inyecta el prompt del sistema y los textos, 
    y extrae la respuesta asegurando un formato JSON estricto.

    Args:
        client (OpenAI): Cliente instanciado de OpenAI.
        modelo (str): Modelo a utilizar (ej. 'gpt-4o').
        system_prompt (str): Instrucciones base para el modelo leídas desde el .txt.
        parrafo (str): El texto de la ley.
        directriz (str): La actividad definitoria.

    Returns:
        Tuple[Optional[str], Optional[int]]: El razonamiento y el valor numérico (1, 0, -1). 
        Devuelve (None, None) en caso de fallo de red o API.
    """
    user_prompt = f"- Párrafo: {parrafo}\n- Directriz: {directriz}"
    
    try:
        response = client.chat.completions.create(
            model=modelo,
            response_format={"type": "json_object"},
            temperature=0,  # Temperatura 0 para maximizar el determinismo (respuestas objetivas)
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        contenido = response.choices[0].message.content
        resultado_json = json.loads(contenido)
        
        razonamiento = resultado_json.get("razonamiento", "Sin razonamiento proporcionado")
        resultado_val = resultado_json.get("resultado")
        
        # Casteo de seguridad: Forzamos la salida del LLM a un número entero nativo de Python
        if resultado_val is not None:
            resultado_val = int(resultado_val)
            
        return razonamiento, resultado_val
        
    except Exception as e:
        print(f"\n  [!] Error de API al procesar un registro: {e}")
        return None, None


def procesar_clasificacion_direccional(
    ruta_csv_entrada: Union[str, Path],
    ruta_prompt: Union[str, Path],
    ruta_csv_salida: Union[str, Path],
    api_key: str,
    modelo: str = "gpt-4o"
) -> bool:
    """
    Lee un archivo CSV con emparejamientos, clasifica la dirección de cada uno mediante
    un LLM y guarda los resultados fila por fila para evitar pérdida de datos ante interrupciones.

    Args:
        ruta_csv_entrada (Union[str, Path]): CSV filtrado (ej. _matches_limpio_).
        ruta_prompt (Union[str, Path]): Archivo .txt con el prompt.
        ruta_csv_salida (Union[str, Path]): CSV resultante (ej. _matches_direccion_).
        api_key (str): Clave de acceso a la API.
        modelo (str): Modelo a utilizar.

    Returns:
        bool: True si el proceso finaliza con éxito, False en caso de error crítico.
    """
    ruta_csv_entrada = Path(ruta_csv_entrada)
    ruta_prompt = Path(ruta_prompt)
    ruta_csv_salida = Path(ruta_csv_salida)

    # 1. Validaciones previas
    if not ruta_prompt.exists():
        print(f"  ✗ Error: No se encuentra el prompt en {ruta_prompt}")
        return False

    with open(ruta_prompt, "r", encoding="utf-8") as f:
        system_prompt = f.read()

    try:
        df_entrada = pd.read_csv(ruta_csv_entrada, sep=';', encoding='utf-8-sig')
    except Exception as e:
        print(f"  ✗ Error al leer {ruta_csv_entrada.name}: {e}")
        return False

    total_filas = len(df_entrada)
    if total_filas == 0:
        print(f"  ⚠ El archivo {ruta_csv_entrada.name} está vacío. Saltando procesamiento.")
        return True

    # 2. Configuración de API y Checkpoint
    client = OpenAI(api_key=api_key)
    filas_procesadas = 0

    if ruta_csv_salida.exists():
        # Si el archivo ya existe, contamos cuántas filas se han analizado previamente
        df_salida = pd.read_csv(ruta_csv_salida, sep=';', encoding='utf-8-sig')
        filas_procesadas = len(df_salida)
        
        if filas_procesadas >= total_filas:
            return True
        else:
            print(f"  > Reanudando proceso: {filas_procesadas} filas previamente completadas.")
    else:
        # Si no existe, creamos el archivo base con las cabeceras exactas
        columnas_base = list(df_entrada.columns) + ["Razonamiento_API", "Dirección"]
        df_vacio = pd.DataFrame(columns=columnas_base)
        df_vacio.to_csv(ruta_csv_salida, index=False, sep=';', encoding='utf-8-sig')

    print(f"  > Procesando las {total_filas - filas_procesadas} filas restantes...")

    # 3. Iteración y guardado continuo (Append)
    # Abrimos el archivo en modo 'a' (append) para añadir líneas sin borrar lo anterior
    with open(ruta_csv_salida, mode='a', encoding='utf-8-sig', newline='') as f_out:
        
        for index in range(filas_procesadas, total_filas):
            fila = df_entrada.iloc[index]
            parrafo = fila["Párrafo de la Ley"]
            directriz = fila["Actividad"]
            
            razonamiento, resultado = analizar_textos_llm(
                client=client, 
                modelo=modelo, 
                system_prompt=system_prompt, 
                parrafo=parrafo, 
                directriz=directriz
            )
            
            # Si la API falla, cortamos la ejecución limpiamente. 
            # Los datos procesados hasta ahora ya están seguros en el disco.
            if razonamiento is None and resultado is None:
                print(f"\n  ✗ Ejecución detenida en la fila {index} por un error de conexión/API.")
                print("    Vuelve a ejecutar la celda más tarde para reanudar desde este punto exacto.")
                return False
            
            # Construimos el registro y lo adjuntamos al CSV
            registro_actual = fila.copy()
            registro_actual["Razonamiento_API"] = razonamiento
            registro_actual["Dirección"] = resultado
            
            # Guardamos solo la fila actual sin incluir cabeceras nuevas
            df_fila = pd.DataFrame([registro_actual])
            df_fila.to_csv(f_out, header=False, index=False, sep=';')
            
            # Feedback visual de progreso
            if (index + 1) % 20 == 0 or (index + 1) == total_filas:
                print(f"    Completado {index + 1} / {total_filas}...")

    print(f"  ✓ Clasificación direccional completada con éxito.")
    return True