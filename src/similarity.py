import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List
from sklearn.metrics.pairwise import cosine_similarity

# =====================================================================
# FUNCIÓN PARA EL CUADERNO 02 (CÁLCULO DE UMBRAL)
# =====================================================================
def calcular_max_similitud(vector_parrafo: List[float], matriz_actividades: np.ndarray) -> float:
    """
    Calcula el coseno de similitud entre un párrafo y todas las actividades
    de una dimensión, devolviendo EXCLUSIVAMENTE el valor máximo.
    (Uso: Fase de validación para determinar umbrales óptimos).

    Args:
        vector_parrafo (List[float]): El vector de embedding de un solo párrafo.
        matriz_actividades (np.ndarray): Matriz 2D con los embeddings de todas las actividades.

    Returns:
        float: El valor de similitud más alto (entre 0 y 1).
    """
    vector_reformado = np.array(vector_parrafo).reshape(1, -1)
    # cosine_similarity devuelve una matriz, np.max extrae el valor más alto
    return float(np.max(cosine_similarity(vector_reformado, matriz_actividades)))


# =====================================================================
# FUNCIÓN PARA EL CUADERNO 03 (ANÁLISIS PRINCIPAL)
# =====================================================================
def procesar_similitudes_ley(
    ruta_ley_pkl: Union[str, Path], 
    ruta_act_pkl: Union[str, Path], 
    umbral: float, 
    ruta_salida: Union[str, Path]
) -> bool:
    """
    Cruza todos los párrafos de una ley contra todas las actividades de una dimensión.
    Extrae TODOS los emparejamientos (matches) que superen o igualen el umbral,
    permitiendo que un párrafo coincida con múltiples actividades.

    Args:
        ruta_ley_pkl (Union[str, Path]): Ruta al .pkl con los embeddings de la ley.
        ruta_act_pkl (Union[str, Path]): Ruta al .pkl con los embeddings de las actividades.
        umbral (float): El umbral de corte óptimo a aplicar.
        ruta_salida (Union[str, Path]): Ruta donde se guardará el CSV resultante.

    Returns:
        bool: True si el proceso fue exitoso, False en caso de error.
    """
    ruta_ley_pkl = Path(ruta_ley_pkl)
    ruta_act_pkl = Path(ruta_act_pkl)
    ruta_salida = Path(ruta_salida)

    try:
        # 1. Cargar los DataFrames
        df_ley = pd.read_pickle(ruta_ley_pkl)
        df_act = pd.read_pickle(ruta_act_pkl)

        # 2. Detección dinámica de la columna de texto (ignora la columna 'Embedding')
        col_texto_ley = [col for col in df_ley.columns if col != "Embedding"][0]
        col_texto_act = [col for col in df_act.columns if col != "Embedding"][0]

        # 3. Apilar los vectores en matrices 2D
        matriz_ley = np.vstack(df_ley["Embedding"].values)
        matriz_actividades = np.vstack(df_act["Embedding"].values)

        # 4. Cálculo de la similitud matricial masiva
        # Devuelve una matriz donde filas = párrafos de la ley, columnas = actividades
        matriz_similitud = cosine_similarity(matriz_ley, matriz_actividades)

        # 5. Filtrado múltiple: Encontrar TODAS las coordenadas que superen el umbral
        filas_ley, cols_act = np.where(matriz_similitud >= umbral)

        resultados = []
        # Iterar sobre las coordenadas encontradas para extraer los textos correspondientes
        for idx_ley, idx_act in zip(filas_ley, cols_act):
            resultados.append({
                "Párrafo de la Ley": df_ley.iloc[idx_ley][col_texto_ley],
                "Actividad": df_act.iloc[idx_act][col_texto_act],
                "Similitud": round(float(matriz_similitud[idx_ley, idx_act]), 4)
            })

        # 6. Guardado y exportación
        df_resultados = pd.DataFrame(resultados, columns=["Párrafo de la Ley", "Actividad", "Similitud"])
        
        if df_resultados.empty:
            print(f"  ⚠ Ningún párrafo superó el umbral de {umbral}.")
            # Creamos un CSV vacío pero con cabeceras para no romper el pipeline de los siguientes cuadernos
            df_resultados.to_csv(ruta_salida, index=False, encoding='utf-8-sig', sep=';')
            return True

        # Ordenar de mayor a menor similitud
        df_resultados = df_resultados.sort_values(by="Similitud", ascending=False)
        
        # Guardar en CSV compatible con Excel europeo (sep=';', utf-8-sig)
        df_resultados.to_csv(ruta_salida, index=False, encoding='utf-8-sig', sep=';')
        
        print(f"  ✓ Generado: {ruta_salida.name} ({len(df_resultados)} emparejamientos encontrados)")
        return True

    except Exception as e:
        print(f"  ✗ Error al procesar similitudes para {ruta_ley_pkl.name}: {e}")
        return False