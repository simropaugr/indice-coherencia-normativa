import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union
from sklearn.metrics import roc_curve, auc, precision_recall_curve

def evaluar_y_visualizar_doble(
    y_real: Union[list, np.ndarray], 
    y_scores: Union[list, np.ndarray], 
    nombre_dimension: str
) -> Tuple[float, float]:
    """
    Evalúa el rendimiento de los cosenos de similitud frente a una clasificación manual,
    calcula los umbrales óptimos para F1-Score y F2-Score, y visualiza las curvas ROC y PR.

    Args:
        y_real (Union[list, np.ndarray]): Valores reales clasificados manualmente (0 o 1).
        y_scores (Union[list, np.ndarray]): Valores continuos predichos (similitud máxima).
        nombre_dimension (str): Nombre de la dimensión para los títulos de las gráficas.

    Returns:
        Tuple[float, float]: (Umbral_F1, Umbral_F2). Si hay un error, devuelve (0.0, 0.0).
    """
    # 1. Normalización de formatos de entrada (Pandas Series a NumPy array)
    y_real = np.array(y_real)
    y_scores = np.array(y_scores)

    # 2. Validación de integridad de clases
    # Las curvas ROC y PR requieren que existan al menos un 1 y un 0 en los datos reales.
    if len(np.unique(y_real)) < 2:
        print(f"✗ Error crítico en '{nombre_dimension}':")
        print("  El dataset etiquetado manualmente debe contener tanto 1s como 0s.")
        print("  Revisa tu clasificación manual en el CSV.")
        return 0.0, 0.0

    # ==========================================
    # 3. CÁLCULO DE MÉTRICAS Y UMBRALES
    # ==========================================
    precision, recall, thresholds_pr = precision_recall_curve(y_real, y_scores)
    
    # Calcular F1-Score (Equilibrio 50/50 entre Precisión y Recall)
    f1_scores = np.divide(
        2 * (precision * recall), 
        (precision + recall), 
        out=np.zeros_like(precision), 
        where=(precision + recall) != 0
    )
    
    # Calcular F2-Score (Penaliza más los Falsos Negativos, priorizando Recall)
    f2_scores = np.divide(
        5 * (precision * recall), 
        (4 * precision + recall), 
        out=np.zeros_like(precision), 
        where=(4 * precision + recall) != 0
    )
    
    # Extraer índices de los valores máximos
    idx_f1 = int(np.argmax(f1_scores))
    idx_f2 = int(np.argmax(f2_scores))
    
    # Extraer los umbrales correspondientes
    umbral_f1 = float(thresholds_pr[idx_f1]) if idx_f1 < len(thresholds_pr) else 0.0
    umbral_f2 = float(thresholds_pr[idx_f2]) if idx_f2 < len(thresholds_pr) else 0.0

    # Calcular métricas para la Curva ROC
    fpr, tpr, _ = roc_curve(y_real, y_scores)
    roc_auc = auc(fpr, tpr)

    # ==========================================
    # 4. VISUALIZACIÓN
    # ==========================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Evaluación de Umbrales: {nombre_dimension}', fontsize=16)

    # Gráfica ROC
    ax1.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC (AUC = {roc_auc:.3f})')
    ax1.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax1.set_xlabel('Tasa de Falsos Positivos')
    ax1.set_ylabel('Tasa de Verdaderos Positivos (Recall)')
    ax1.set_title('Curva ROC')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)

    # Gráfica Precision-Recall
    ax2.plot(recall, precision, color='blue', lw=2, label='Curva PR')
    
    # Punto óptimo F1 (Rojo)
    ax2.scatter(recall[idx_f1], precision[idx_f1], color='red', s=100, zorder=5, 
                label=f'Max F1 (Umbral: {umbral_f1:.3f})')
    
    # Punto óptimo F2 (Verde)
    ax2.scatter(recall[idx_f2], precision[idx_f2], color='green', s=100, zorder=5, 
                label=f'Max F2 (Umbral: {umbral_f2:.3f})')
    
    ax2.set_xlabel('Exhaustividad (Recall)')
    ax2.set_ylabel('Precisión (Precision)')
    ax2.set_title('Curva Precision-Recall (Comparativa)')
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show() # Renderiza los gráficos en el cuaderno Jupyter

    # ==========================================
    # 5. REPORTE EN CONSOLA
    # ==========================================
    print(f"--- RESULTADOS: {nombre_dimension.upper()} ---")
    print(">> ESTRATEGIA 1: EQUILIBRIO (Máximo F1-Score)")
    print(f"   Umbral Sugerido:       {umbral_f1:.4f}")
    print(f"   Precisión en ese punto:{precision[idx_f1]:.4f}")
    print(f"   Recall en ese punto:   {recall[idx_f1]:.4f}\n")
    
    print(">> ESTRATEGIA 2: PRIORIZAR RECALL (Máximo F2-Score)")
    print(f"   Umbral Sugerido:       {umbral_f2:.4f}")
    print(f"   Precisión en ese punto:{precision[idx_f2]:.4f}")
    print(f"   Recall en ese punto:   {recall[idx_f2]:.4f}")
    print("-" * 50 + "\n")

    return umbral_f1, umbral_f2