# Entregable 5

## Datos generales
Fecha de entrega: Domingo 15 de Noviembre.

## Objetivo
* Implementar el monitoreo de los datos del modelo del proyecto MiBici con Evidently.
* Detectar, interpretar y reportar el drift entre los datos de entrenamiento y datos de un periodo posterior.
* Comparar dos versiones del modelo registradas en MLFlow mediante un ejercicio de A/B testing, para decidir con evidencia cuál ofrece mejor desempeño.
* Esta actividad corresponde a una fase de entrega del proyecto del curso (Entregable 8): el equipo añade al proyecto la capa de observabilidad que permite mantener el modelo vigente a lo largo del tiempo.

* Seguiremos trabajando con el proyecto MiBici, construido a partir de los datos abiertos de MiBici: esta entrega se construye sobre el mismo repositorio del equipo y conserva el trabajo de las entregas anteriores (modelo registrado, pruebas y API en contenedor). Trabajen con al menos uno de los modelos del proyecto (regresión, clasificación o clustering).


## Instrucciones
En esta actividad integradora, como equipo, dotarán al proyecto de la capacidad de vigilarse a sí mismo. El objetivo es detectar cuándo los datos de producción se alejan de los de entrenamiento y decidir, con evidencia, si una nueva versión del modelo es mejor que la actual.

Realiza, junto con tu equipo, los siguientes pasos:

1. Prepara los conjuntos de datos: Definan un conjunto de referencia (el periodo de datos de MiBici con el que se entrenó el modelo) y un conjunto actual que represente el comportamiento en producción: un periodo posterior de los datos abiertos de MiBici (otro mes, temporada o año). Si su API de la semana 8 registra las peticiones que recibe, también pueden usar esos registros. Documenten qué periodos eligieron y por qué.
2. Genera el reporte de drift con Evidently: Utilicen Evidently para comparar ambos conjuntos y producir un reporte que muestre el data drift por variable y la calidad de los datos.
3. Interpreta los resultados: Analicen el reporte e identifiquen qué variables presentan drift significativo y qué implicaciones tendría para el desempeño del modelo.
4. Entrena una versión alternativa: Generen una segunda versión del modelo (por ejemplo, reentrenada incluyendo datos del periodo actual o con distintos hiperparámetros) y regístrenla en el Model Registry de MLFlow como una nueva versión del mismo modelo.
5. Realiza el A/B testing: Comparen las dos versiones del modelo sobre el mismo conjunto de evaluación del periodo actual, contrastando sus métricas, y determinen de forma justificada cuál debería quedar como versión vigente.
6. Elabora el reporte: Integren en un documento el reporte de drift (con gráficas de Evidently), el resultado del A/B testing y una recomendación sobre si convendría reentrenar el modelo (Continuous Training).


## Criterios de Evaluación
La actividad se evaluará con base en la siguiente rúbrica (100 puntos en total):

| Criterio | Excelente (100%) | Satisfactorio (70%) | Insuficiente (<50%) | Peso |
|---|---|---|---|:---:|
Generación del reporte de drift con Evidently (conjuntos de referencia y actual bien definidos, reporte correctamente producido). | Define con claridad y precisión los conjuntos de referencia y actual. Genera un reporte completo y estructurado con Evidently, incluyendo gráficas y métricas relevantes. | Define los conjuntos de referencia y actual de manera adecuada, aunque con ligeros errores o ambigüedades. El reporte es funcional, pero incompleto o poco claro. | Presenta conjuntos de datos mal definidos o ambiguos. El reporte es deficiente, con información incompleta, incorrecta o ausente. | 30% |
Interpretación del drift (identificación de variables afectadas e implicaciones para el modelo). | Identifica con precisión las variables afectadas por el drift. Analiza de manera detallada las implicaciones para el desempeño del modelo, mostrando un razonamiento profundo. | Identifica las variables afectadas, pero con análisis parcial o limitado. Las implicaciones son mencionadas de forma general, sin suficiente profundidad. | Omite identificar variables clave o realiza un análisis superficial que no permite extraer conclusiones claras sobre las implicaciones del drift. | 20% |
A/B testing entre versiones (comparación metodológicamente correcta de dos versiones del modelo).| Realiza una comparación metodológicamente sólida y bien justificada entre las dos versiones del modelo, detallando métricas y resultados con claridad.| Realiza la comparación entre versiones con metodología adecuada, aunque con errores menores o explicaciones poco claras. | Realiza una comparación incompleta, con errores metodológicos o sin justificar adecuadamente las diferencias entre las versiones. | 30% |
Recomendación y razonamiento sobre Continuous Training (decisión justificada con base en la evidencia). | Presenta una recomendación clara y bien fundamentada, respaldada por los datos y el análisis realizado, demostrando un razonamiento crítico. | Presenta una recomendación adecuada, pero con fundamentos o justificaciones limitadas o ambiguas. | La recomendación es vaga, sin justificación suficiente o basada en un análisis incompleto o incorrecto. | 10% |
Claridad del reporte y trabajo colaborativo evidenciado en Git. | El reporte es claro, bien estructurado y profesional. Evidencia claramente el trabajo colaborativo en Git, con contribuciones balanceadas y organizadas. | El reporte es adecuado, pero presenta problemas menores de claridad o estructura. La evidencia en Git muestra participación desigual o desorganizada. | El reporte es confuso o desorganizado. La evidencia en Git es insuficiente o no refleja una colaboración efectiva entre los integrantes del equipo. | 10% |