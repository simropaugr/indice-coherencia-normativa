import csv
import pymupdf  # PyMuPDF
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional

def convertir_pdf_a_markdown(ruta_pdf: Union[str, Path], ruta_md: Union[str, Path]) -> bool:
    """
    Abre un archivo PDF, extrae su texto agrupado por bloques visuales,
    limpia los saltos de línea internos y lo exporta a formato Markdown.

    Args:
        ruta_pdf (Union[str, Path]): Ruta de origen del archivo PDF.
        ruta_md (Union[str, Path]): Ruta de destino para el archivo Markdown.

    Returns:
        bool: True si la conversión fue exitosa, False en caso de error.
    """
    ruta_pdf = Path(ruta_pdf)
    ruta_md = Path(ruta_md)

    print(f"Procesando: {ruta_pdf.name}...")

    if not ruta_pdf.exists():
        print(f"✗ Error: El archivo de origen '{ruta_pdf}' no existe.")
        return False

    try:
        ruta_md.parent.mkdir(parents=True, exist_ok=True)
        texto_markdown = ""

        with pymupdf.open(ruta_pdf) as documento:
            for pagina in documento:
                bloques = pagina.get_text("blocks")
                for bloque in bloques:
                    texto_bloque = bloque[4].strip()
                    if texto_bloque:
                        parrafo_limpio = texto_bloque.replace("\n", " ")
                        texto_markdown += parrafo_limpio + "\n\n"

        with open(ruta_md, "w", encoding="utf-8") as archivo:
            archivo.write(texto_markdown)

        print(f"✓ Éxito. Guardado en: {ruta_md}\n")
        return True

    except Exception as e:
        print(f"✗ Error inesperado al procesar '{ruta_pdf.name}': {e}\n")
        return False


def convertir_md_a_csv(
    ruta_md: Union[str, Path], 
    ruta_csv: Union[str, Path], 
    columnas_extra: Optional[List[str]] = None
) -> bool:
    """
    Lee un archivo Markdown, separa su contenido en párrafos y lo exporta 
    a un archivo CSV preparado para la clasificación manual.

    Args:
        ruta_md (Union[str, Path]): Ruta de origen del archivo Markdown.
        ruta_csv (Union[str, Path]): Ruta de destino para el archivo CSV.
        columnas_extra (Optional[List[str]]): Lista de nombres para las columnas 
            de clasificación manual. Si es None, usa valores por defecto.

    Returns:
        bool: True si la conversión fue exitosa, False en caso de error.
    """
    ruta_md = Path(ruta_md)
    ruta_csv = Path(ruta_csv)

    print(f"Procesando: {ruta_md.name}...")

    if not ruta_md.exists():
        print(f"✗ Error: El archivo de origen '{ruta_md}' no existe.")
        return False

    if columnas_extra is None:
        columnas_extra = ["Similitud_Dimension1", "Similitud_Dimension2"]

    try:
        ruta_csv.parent.mkdir(parents=True, exist_ok=True)

        with open(ruta_md, "r", encoding="utf-8") as archivo_md:
            contenido = archivo_md.read()

        parrafos = [p.strip() for p in contenido.split("\n\n") if p.strip()]

        if not parrafos:
            print(f"⚠ Advertencia: El archivo '{ruta_md.name}' está vacío o no contiene párrafos válidos.")
            return False

        with open(ruta_csv, "w", encoding="utf-8-sig", newline='') as archivo_csv:
            escritor = csv.writer(archivo_csv, delimiter= ';',quoting=csv.QUOTE_ALL)
            
            cabecera = ["ID_Parrafo", "Texto"] + columnas_extra
            escritor.writerow(cabecera)
            
            for i, parrafo in enumerate(parrafos, 1):
                fila = [i, parrafo] + [""] * len(columnas_extra)
                escritor.writerow(fila)

        print(f"✓ Éxito. Exportados {len(parrafos)} párrafos en: {ruta_csv}\n")
        return True

    except Exception as e:
        print(f"✗ Error inesperado al procesar '{ruta_md.name}': {e}\n")
        return False


def load_labeled_data(ruta_csv: Union[str, Path]) -> Optional[pd.DataFrame]:
    """
    Carga un archivo CSV clasificado manualmente, manejando automáticamente
    los problemas comunes de codificación y delimitadores generados por Excel.

    Args:
        ruta_csv (Union[str, Path]): Ruta al archivo CSV etiquetado por el usuario.

    Returns:
        Optional[pd.DataFrame]: DataFrame con los datos cargados o None si hay error.
    """
    ruta_csv = Path(ruta_csv)
    
    print(f"Cargando datos etiquetados desde: {ruta_csv.name}...")

    if not ruta_csv.exists():
        print(f"✗ Error: El archivo '{ruta_csv}' no existe. Asegúrate de haberlo guardado en la carpeta correcta.")
        return None

    try:
        df = pd.read_csv(ruta_csv, sep= ';',encoding="utf-8")
        if df.shape[1] == 1:
            df = pd.read_csv(ruta_csv, encoding="utf-8", sep=";")
            
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(ruta_csv, encoding="latin1", sep=";")
        except Exception as e:
            print(f"✗ Error al decodificar el archivo en formato Windows: {e}")
            return None
    except Exception as e:
        print(f"✗ Error inesperado al leer el archivo: {e}")
        return None

    columnas_requeridas = ["ID_Parrafo", "Texto"]
    if not all(col in df.columns for col in columnas_requeridas):
        print(f"✗ Error: El archivo ha sido corrompido. Faltan las columnas obligatorias: {columnas_requeridas}")
        return None

    print(f"✓ Éxito. Dataset cargado con {df.shape[0]} filas y {df.shape[1]} columnas.\n")
    return df