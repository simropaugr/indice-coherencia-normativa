import os
import pandas as pd
from openai import OpenAI
from pathlib import Path
from typing import Union, Optional

def calcular_embeddings_dataframe(
    df: pd.DataFrame, 
    columna_texto: str, 
    modelo: str = "text-embedding-3-small"
) -> pd.DataFrame:
    """
    Calcula los embeddings para una columna específica de un DataFrame utilizando
    la API de OpenAI, procesando los datos en lotes para optimizar tiempos y evitar bloqueos.

    Args:
        df (pd.DataFrame): DataFrame original.
        columna_texto (str): Nombre de la columna que contiene los textos a vectorizar.
        modelo (str): Modelo de OpenAI a utilizar.

    Returns:
        pd.DataFrame: Un nuevo DataFrame con la columna 'Embedding' añadida.
    """
    # 1. Instanciación del cliente (asume que OPENAI_API_KEY ya está en os.environ)
    client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
    
    textos = df[columna_texto].tolist()
    embeddings = []
    tamaño_lote = 100 
    
    print(f"  > Calculando embeddings para {len(textos)} textos en lotes de {tamaño_lote}...")

    # 2. Procesamiento por lotes (Batching)
    for i in range(0, len(textos), tamaño_lote):
        lote = textos[i : i + tamaño_lote]
        
        try:
            respuesta = client.embeddings.create(
                input=lote,
                model=modelo
            )
            # Extraemos los vectores en el orden en que se enviaron
            embeddings.extend([datos.embedding for datos in respuesta.data])
            print(f"    Completado {min(i + tamaño_lote, len(textos))}/{len(textos)}...")
            
        except Exception as e:
            print(f"    ✗ Error crítico en el lote {i} a {i + tamaño_lote}: {e}")
            raise  # Detenemos la ejecución en caso de fallo de API para evitar datos corruptos

    # 3. Integración estructurada
    # Usamos .copy() para evitar SettingWithCopyWarning de Pandas
    df_resultado = df.copy()
    df_resultado["Embedding"] = embeddings
    
    return df_resultado


def procesar_markdown_a_embeddings(
    ruta_md: Union[str, Path], 
    ruta_pkl: Union[str, Path], 
    modelo: str = "text-embedding-3-small"
) -> Optional[pd.DataFrame]:
    """
    Lee un archivo Markdown, separa los párrafos, calcula sus embeddings 
    y guarda el resultado estructural en un archivo Pickle.

    Args:
        ruta_md (Union[str, Path]): Ruta del archivo Markdown de entrada.
        ruta_pkl (Union[str, Path]): Ruta de salida para el archivo Pickle.
        modelo (str): Modelo de OpenAI a utilizar.

    Returns:
        Optional[pd.DataFrame]: DataFrame con los resultados o None si hay error.
    """
    ruta_md = Path(ruta_md)
    ruta_pkl = Path(ruta_pkl)

    # 1. Lectura y separación del texto
    with open(ruta_md, 'r', encoding='utf-8') as archivo:
        texto_completo = archivo.read()
    
    parrafos = [p.strip() for p in texto_completo.split('\n\n') if p.strip()]
    
    if not parrafos:
        print(f"  ⚠ Advertencia: El documento '{ruta_md.name}' está vacío. Saltando...")
        return None
        
    print(f"Iniciando: {ruta_md.name} -> Se han encontrado {len(parrafos)} párrafos.")
    
    # 2. Transformación a DataFrame
    df_temporal = pd.DataFrame({
        "Texto": parrafos
    })
    
    # 3. Llamada a la función principal de embeddings (Reutilización de código)
    df_con_embeddings = calcular_embeddings_dataframe(
        df=df_temporal, 
        columna_texto="Texto", 
        modelo=modelo
    )
    
    # 4. Guardado estructural
    df_con_embeddings.to_pickle(ruta_pkl)
    print(f"✓ Éxito. Datos guardados en: {ruta_pkl}\n")
    
    return df_con_embeddings