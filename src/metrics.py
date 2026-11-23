import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union

def calcular_indices_finales(
    ruta_csv_entrada: Union[str, Path], 
    ruta_csv_salida: Union[str, Path], 
    valor_actividades: int, 
    valor_umbral: float
) -> bool:
    """
    Calcula el índice final de un documento en función de la similitud y 
    la dirección obtenida por el LLM, aplicando tus operaciones algebraicas.

    Args:
        ruta_csv_entrada (Union[str, Path]): Ruta del CSV con las direcciones calculadas.
        ruta_csv_salida (Union[str, Path]): Ruta para guardar el CSV final con los índices.
        valor_actividades (int): Número de actividades definitorias para la dimensión.
        valor_umbral (float): Umbral de corte matemático validado.

    Returns:
        bool: True si el proceso fue exitoso, False en caso de error.
    """
    ruta_csv_entrada = Path(ruta_csv_entrada)
    ruta_csv_salida = Path(ruta_csv_salida)

    try:
        # Lectura con parámetros estrictos para evitar corrupción de decimales
        df = pd.read_csv(ruta_csv_entrada, sep=';', encoding='utf-8-sig', decimal='.')
    except Exception as e:
        print(f"  ✗ Error al leer {ruta_csv_entrada.name}: {e}")
        return False

    columnas_req = {'Dirección', 'Similitud'}
    if not columnas_req.issubset(df.columns):
        print(f"  ✗ Error: Faltan las columnas {columnas_req} en el archivo. Se omite.")
        return False

    try:
        # 1. Barrera de seguridad: Garantizar tipos numéricos (int/float)
        # Si encuentra un texto corrupto, coerce lo convierte a NaN, y fillna(0) lo neutraliza
        df['Dirección'] = pd.to_numeric(df['Dirección'], errors='coerce').fillna(0)
        df['Similitud'] = pd.to_numeric(df['Similitud'], errors='coerce').fillna(0)

        # ---------------------------------------------------------------------
        # 2. OPERACIONES MATEMÁTICAS
        # ---------------------------------------------------------------------
        
        # Operación 1: Multiplicación de Dirección por Similitud (redondeado a 4 decimales)
        df['Puntuación'] = (df['Dirección'] * df['Similitud']).round(4)

        # Operación 2: Cuadrado conservando el signo original
        df['Puntuación_al_cuadrado'] = ((df['Puntuación'] ** 2) * np.sign(df['Puntuación'])).round(4)

        # Operación 3: Cálculo del Índice global del documento
        numerador = df['Puntuación_al_cuadrado'].sum()
        denominador = valor_actividades * valor_umbral
        
        # Prevención de división por cero
        if denominador == 0:
            print("  ✗ Error: El denominador (actividades * umbral) es 0. División imposible.")
            return False

        indice_calculado = numerador / denominador
        df['Índice'] = round(indice_calculado, 4)

        # ---------------------------------------------------------------------
        # 3. EXPORTACIÓN
        # ---------------------------------------------------------------------
        df.to_csv(ruta_csv_salida, sep=';', encoding='utf-8-sig', index=False, decimal='.')
        
        print(f"  ✓ Índice global de la ley obtenido: {indice_calculado:.4f}")
        print(f"  ✓ Datos matemáticos guardados en: {ruta_csv_salida.name}")
        return True

    except Exception as e:
        print(f"  ✗ Error en las operaciones matemáticas para {ruta_csv_entrada.name}: {e}")
        return False