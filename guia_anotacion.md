# Guía de Anotación: Scoring de Compatibilidad Candidato-Oferta

## 1. Objetivo

El objetivo de esta tarea es clasificar la compatibilidad entre un perfil de candidato (CV) y una oferta de trabajo. Vuestra anotación creará el conjunto de datos "Gold Standard" que se usará para entrenar y evaluar nuestro sistema de inteligencia artificial.

La consistencia entre anotadores es el factor más importante. Por favor, seguid esta guía rigurosamente.

---

## 2. La Escala de Evaluación (5 Categorías)

Deberéis asignar a cada par (CV, Oferta) una de las siguientes cinco categorías. Junto a cada categoría se muestra el rango numérico que el sistema usará internamente.

| Categoría       | Rango Numérico | Significado                                                                 |
|-----------------|----------------|------------------------------------------------------------------------------|
| 🟢 STRONG YES   | 85-100         | Un encaje excepcional. El candidato ideal que querrías contratar inmediatamente. |
| 🟡 YES          | 70-84          | Un buen candidato. Cumple los requisitos y debería pasar a una entrevista.       |
| 🟠 MAYBE        | 50-69          | Un candidato dudoso o en el límite. Podría valer la pena considerarlo si faltan opciones. |
| 🔴 WEAK NO      | 30-49          | Un mal encaje. Probablemente no vale la pena invertir tiempo en este candidato. |
| ⚫ STRONG NO    | 0-29           | Un descarte definitivo. El perfil es irrelevante para la oferta.               |

---

## 3. Criterios Detallados por Categoría

Usa estos criterios para guiar tu decisión.

### 🟢 STRONG YES (Contratar Inmediatamente)
- **Cumplimiento Total**: El candidato cumple todos los requisitos marcados como "obligatorios" o "imprescindibles" en la oferta.
- **Valor Añadido**: Cumple también con varios de los requisitos "deseables", "plus" o "valorables".
- **Experiencia Relevante**: La experiencia laboral es en el mismo sector, dominio o con un tipo de producto/cliente muy similar. Parece un perfil "plug-and-play".

### 🟡 YES (Entrevistar)
- **Cumplimiento Obligatorio**: El candidato cumple la gran mayoría (si no todos) los requisitos obligatorios.
- **Experiencia Sólida**: La experiencia laboral es claramente relevante, aunque quizás no sea en un sector idéntico. Las habilidades y responsabilidades son transferibles.
- **Potencial**: Aunque falte algún requisito deseable, el perfil muestra un claro potencial para aprenderlo rápidamente y tener éxito en el puesto.

### 🟠 MAYBE (En el Límite)
- **Perfil Mixto**: El candidato cumple algunos requisitos obligatorios, pero falla en otros que son importantes.
- **Experiencia Tangencial**: La experiencia es en un dominio diferente, pero se podrían argumentar algunas similitudes. Por ejemplo, un "Data Analyst" aplicando a un puesto de "BI Developer".
- **Compensación**: Falla en un área (ej. años de experiencia) pero lo compensa en otra (ej. dominio de una tecnología muy específica y demandada). También se incluyen aquí los perfiles sobrecualificados que podrían rechazar la oferta por no ser un reto.

### 🔴 WEAK NO (Probablemente No)
- **Incumplimiento Clave**: Faltan requisitos obligatorios fundamentales para el desempeño del puesto (ej. un idioma requerido, una certificación legal, etc.).
- **Experiencia Insuficiente**: No cumple con los años mínimos de experiencia o sus responsabilidades pasadas no se alinean con las de la oferta.
- **Habilidades Ajenas**: Las tecnologías y habilidades listadas en el CV tienen poca o ninguna relación con las solicitadas en la oferta.

### ⚫ STRONG NO (Definitivamente No)
- **Sin Coincidencias**: Prácticamente no hay ninguna coincidencia entre los requisitos de la oferta y el perfil del candidato.
- **Dominio Irrelevante**: El CV pertenece a una categoría profesional completamente distinta (ej. un chef aplicando a un puesto de programador).
- **Descarte Evidente**: La candidatura es claramente un error o spam.

---

## 4. Principios Generales y Reglas de Decisión

- **Prioriza la Experiencia y las Habilidades Técnicas**: A menos que la oferta especifique un título universitario como requisito legal o indispensable, da más peso a la experiencia laboral demostrable y a las habilidades técnicas ("hard skills").
- **Sé un Reclutador Pragmático**: Pregúntate: "¿Podría esta persona empezar a aportar valor real en las primeras semanas?". Esto ayuda a diferenciar entre un YES y un MAYBE.
- **En caso de duda, sé conservador**: Si dudas entre dos categorías (ej. entre YES y MAYBE), elige siempre la más baja (MAYBE). Esto nos ayuda a calibrar el modelo de forma más prudente.

---

## 5. Formato de Entrega

Por favor, entrega tus anotaciones en un archivo **CSV** con tres columnas:
| id_oferta       | id_cv | categoría       