# Reporte de Implementación del Modelo Baseline

**Fecha:** `YYYY-MM-DD`
**Autor:** `Tu Nombre`
**ID del Commit:** `(opcional, el hash de git del commit actual)`

## 1. Descripción del Baseline

El sistema baseline sirve como punto de referencia simple pero robusto para medir el rendimiento del modelo principal, DistilMatch. El objetivo de cualquier modelo más complejo es superar significativamente las métricas de este baseline.

La implementación se basa en una arquitectura clásica de **recuperación de información semántica**:
1.  **Representación de Texto:** Tanto la oferta de trabajo como el CV del candidato se convierten en un único string estructurado utilizando las funciones de `src.text_utils`.
2.  **Generación de Embeddings:** Se utiliza un modelo pre-entrenado de la librería `sentence-transformers` para convertir cada string de texto en un vector numérico de alta dimensión (embedding).
3.  **Cálculo de Similitud:** La compatibilidad entre la oferta y el CV se calcula usando la **similitud del coseno** entre sus respectivos vectores de embedding. El score resultante es un valor entre -1 y 1 (generalmente entre 0 y 1 para textos).

## 2. Implementación Técnica

-   **Modelo de Embeddings:** Se ha seleccionado el modelo `all-MiniLM-L6-v2`. Se eligió por su excelente equilibrio entre rendimiento y velocidad, lo que lo hace ideal para un baseline rápido.
-   **Librerías Clave:** `sentence-transformers`, `pandas`.
-   **Ubicación del Código:** La implementación y las pruebas se encuentran en el notebook: `notebooks/02_baseline_implementation.ipynb`.

## 3. Ejemplo de Funcionamiento

A continuación se muestra un ejemplo de ejecución con un par de muestra:

-   **Oferta de ejemplo (ID):** `(Pega el ID de la oferta de tu notebook)`
-   **CV de ejemplo (ID):** `(Pega el ID del candidato de tu notebook)`
-   **Score de Similitud del Coseno Obtenido:** `(Pega el score de tu notebook)`

## 4. Evaluación Formal de Rendimiento

**(PENDIENTE)**

*Esta sección se completará una vez que el `gold_standard_test_set.csv` esté finalizado. El baseline se ejecutará sobre todo el conjunto de test y se calcularán las siguientes métricas comparando sus scores con las anotaciones humanas:*

-   **Métricas de Regresión:**
    -   Correlación de Pearson (r)
    -   Error Absoluto Medio (MAE)
    -   Raíz del Error Cuadrático Medio (RMSE)
-   **Métricas de Ranking:**
    -   Correlación de Rango de Spearman (ρ)
