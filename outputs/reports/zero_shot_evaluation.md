# Informe de Evaluación Zero-Shot (En Progreso)

## Fase 1: Baseline del Modelo Student

- **Modelo Evaluado:** `Qwen/Qwen2-7B-Instruct`
- **Fecha:** 2024-XX-XX
- **Conjunto de Datos:** `gold_standard_test.jsonl`

### Resultados Cuantitativos

| Métrica                    | Valor   |
|----------------------------|---------|
| Error Absoluto Medio (MAE) | 27.1591 |
| Correlación de Spearman (ρ)| 0.3878  |

### Análisis Cualitativo Inicial

Los resultados del baseline muestran que el modelo "de fábrica" tiene una comprensión básica pero insuficiente de la tarea. Un MAE de ~27 puntos indica que el modelo frecuentemente se equivoca en una categoría entera de la rúbrica. La correlación de Spearman de ~0.39, aunque positiva, es débil, lo que sugiere que su capacidad para rankear candidatos de forma fiable es limitada.

Este rendimiento justifica plenamente la necesidad de un fine-tuning especializado para mejorar la precisión y la fiabilidad del modelo.