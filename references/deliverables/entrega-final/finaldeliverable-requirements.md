# Construye | Entrega Proyecto Final | Colaborativo

## Objetivos
Integrar en un solo proyecto de extremo a extremo todas las competencias desarrolladas en el curso, entregando un sistema de Machine Learning completo, reproducible y operable: desde la estructura del repositorio y el seguimiento de experimentos, hasta el despliegue como servicio, la automatización con CI/CD, el monitoreo del modelo en producción y la documentación responsable del mismo.

Esta actividad constituye la entrega final del proyecto integrador del curso: la demostración de que el equipo es capaz de llevar un modelo de Machine Learning desde una idea hasta un sistema listo para el mundo real, aplicando las prácticas de MLOps.

## Instrucciones
En esta actividad final, como equipo consolidarán todo el trabajo realizado a lo largo del curso en un proyecto integrador coherente. La evidencia de su funcionamiento será un video grabado por el equipo; no habrá presentación en vivo ante el grupo. El objetivo es demostrar el dominio del ciclo de vida completo de MLOps sobre el proyecto que han venido construyendo semana a semana.

Realiza, junto con tu equipo, los siguientes pasos:

1. Consolida el repositorio: Asegúrense de que el proyecto integre, de forma ordenada y reproducible, la estructura (Cookiecutter), el feature engineering, el código refactorizado como MLFlow Project y el seguimiento de experimentos con MLFlow Tracking.
2. Verifica el modelo en el Registry: Confirmen que el mejor modelo esté registrado y versionado en el Model Registry, con su linaje y su estado de producción bien definidos.
3. Integra el despliegue y la operación: Reúnan la API (FastAPI) contenerizada, la orquestación con Docker Compose, el pipeline de CI/CD con GitHub Actions y las pruebas de carga con Locust.
4. Incluye el monitoreo y la gobernanza: Incorporen el reporte de drift con Evidently, el A/B testing entre versiones, el análisis de FinOps y el Model Card que documenta el modelo.
5. Prepara la presentación: Elaboren una presentación que sintetice el problema, la solución y las decisiones de MLOps tomadas.
6. Graba el video de evidencia: Graben un video de mínimo 2 y máximo 5 minutos que recorra la presentación y muestre una demo funcional del sistema (por ejemplo, la API respondiendo a una petición real).

Valor agregado (opcional): si el equipo lo desea, puede mostrar un ciclo de Continuous Training, es decir, un re-entrenamiento disparado por la detección de drift y la promoción de la nueva versión en el Model Registry. Este elemento no es obligatorio ni modifica la rúbrica.

## Especificaciones de Entrega
Para considerar completa la entrega, asegúrate de cumplir con lo siguiente:

* Modalidad: Colaborativa (en equipo de 4 personas).
    * La evaluación se asigna a cada integrante con base en el entregable del equipo.
    * Se recomienda que todos los integrantes contribuyan al repositorio y que su participación sea visible en el historial de Git (commits, pull requests o revisiones de código).
* Medio de realización/entrega: Enlace al repositorio de GitHub del equipo (proyecto completo y funcional), más la presentación final y el video de evidencia (de 2 a 5 minutos), subidos a la plataforma Canvas.
    * Graba un video de mínimo 2 y máximo 5 minutos que muestre los puntos anteriormente solicitados. El video es la evidencia de la presentación; no habrá presentación en vivo.
    * El video es continuo (sin cortes de edición) y lo graban en equipo: un integrante maneja el teclado, los demás pueden aparecer en cámara o narrar por turnos.
* Formato: Repositorio de GitHub con el proyecto integrador completo (código, MLFlow, Docker Compose, CI/CD, Locust, Evidently y Model Card), README general del proyecto, y presentación (por ejemplo, en PDF o diapositivas). La presentación debe cubrir el problema, la solución y las decisiones de MLOps.
* Nombre del entregable:
    1) El repositorio: TC5061-Proyecto-Equipo[Número],
    2) La presentación: TC5061_Proyecto_Final_Equipo[Número].pdf.
    Por ejemplo: TC5061_Proyecto_Final_Equipo03.pdf.
    3) El video explicativo: TC5061_Proyecto_Final_Equipo[Número].mp4 
    O dentro de la presentación se puede integrar el enlace al video sin dificultad de acceso. 

## Criterios de Evaluación
La actividad se evaluará con base en la siguiente rúbrica (100 puntos en total):

| Criterio | Excelente (100%) |  Satisfactorio (70%) | Insuficiente (<=50%) | Peso |
| :--- | :--- | :--- | :--- | :--- |
| Estructura y experimentación: El repositorio está bien estructurado y reproducible, con feature engineering, MLFlow Project y seguimiento de experimentos. | Organiza el repositorio de manera ordenada y reproducible, integrando correctamente la estructura Cookiecutter, el feature engineering, el código refactorizado como MLFlow Project y el seguimiento de experimentos con MLFlow Tracking. | Estructura el repositorio de forma funcional y reproducible, aunque presenta ligeros problemas en la integración de uno o más elementos (por ejemplo, feature engineering o MLFlow Tracking). | El repositorio es funcional, pero carece de organización clara o no integra completamente alguno de los elementos requeridos. | 20% |
| Modelo y Registry: El mejor modelo está correctamente registrado, versionado y documentado en el Model Registry. | Registra y versiona el mejor modelo en el Model Registry, documentando el linaje y definiendo correctamente su estado de producción. | Registra y versiona el modelo en el Model Registry, pero la documentación del linaje o el estado de producción presenta inconsistencias menores. | Registra el modelo en el Model Registry, pero no asegura una documentación clara o el estado de producción está incompleto. | 15% |
| Despliegue y operación: La API contenerizada, Docker Compose, el CI/CD y las pruebas de carga funcionan de forma integrada. | Integra y opera correctamente la API contenerizada, Docker Compose, el pipeline de CI/CD con GitHub Actions y las pruebas de carga con Locust, asegurando su funcionamiento total. | Integra los componentes de despliegue y operación, aunque presenta problemas menores en la funcionalidad o en la integración de alguno de los elementos. | Implementa algunos elementos del despliegue y operación, pero no asegura una integración completa o funcionalidad consistente. | 25% |
| Monitoreo, gobernanza y FinOps: El proyecto incluye reporte de drift, A/B testing, análisis FinOps y Model Card. | Incorpora reporte de drift con Evidently, A/B testing, análisis de FinOps y Model Card, asegurando una documentación completa y prácticas responsables de gobernanza. | Integra los elementos de monitoreo, gobernanza y FinOps, pero con documentación parcial o con aspectos técnicos que requieren ajustes menores. | Incluye algunos elementos de monitoreo, gobernanza o FinOps, pero no asegura una implementación completa ni documentación adecuada. | 20% |
| Presentación y demostración [Video]: La demo es funcional y la presentación comunica con claridad el problema, la solución y las decisiones de MLOps. (Duración de 2 a 5 minutos). | Realiza una demo funcional del sistema, mostrando una API operativa y respondiendo a peticiones reales. La presentación (video de 2 a 5 minutos) sintetiza de forma clara el problema, la solución y las decisiones de MLOps tomadas. | Presenta una demo funcional del sistema, pero la API o la presentación carecen de claridad o consistencia en algunos aspectos. | La demo no muestra la API respondiendo a peticiones reales, la presentación no comunica con claridad el problema, la solución o las decisiones de MLOps, o el video dura menos de 2 minutos o más de 5 minutos. | 20% | 