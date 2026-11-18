# Entregable 6

## Datos generales
Fecha de entrega: Domingo 22 de Noviembre.

## Objetivos
 
Consolidar el proyecto integrador del equipo como un sistema operable:

* Orquestar sus servicios con Docker Compose.
* Automatizar pruebas y despliegue con un pipeline de CI/CD en GitHub Actions.
* Evaluar su desempeño bajo carga con Locust.
* Realizar un análisis de FinOps que estime el costo por inferencia del sistema.
Esta actividad corresponde a una fase de entrega del proyecto del curso (Entregable 7): a partir de la API contenerizada de la Semana 8 y de las versiones del modelo registradas en MLFlow durante la Semana 9, el equipo construye la arquitectura completa de operación, la automatiza y la analiza desde la perspectiva de desempeño y costo.

## Instrucciones
En esta actividad, como equipo, transformarán la API contenerizada en un sistema completo, automatizado y analizado. El objetivo es que el proyecto se levante con un solo comando, se pruebe y despliegue de forma automática, y que el equipo comprenda cómo se comporta bajo carga y cuánto cuesta operarlo.

Realiza, junto con tu equipo, los siguientes pasos:

1. Orquesta los servicios con Docker Compose: Creen un archivo docker-compose.yml que levante los servicios del proyecto (por ejemplo, la API del modelo y el servidor de MLFlow) y que permita ejecutarlos juntos con un solo comando. Pueden usar Docker Compose o podman-compose. Agregar Redis y un worker de RQ para procesar predicciones de forma asíncrona es opcional.
2. Configura el pipeline de CI/CD: Implementen un workflow de GitHub Actions que, al hacer push o pull request, ejecute automáticamente las pruebas del proyecto y construya la imagen del servicio.
3. Ejecuta pruebas de carga con Locust: Diseñen un escenario de prueba que simule múltiples usuarios haciendo peticiones al endpoint de predicción, y generen un reporte con gráficas de desempeño (latencia, peticiones por segundo, tasa de error).
4. Monitorea el uso de recursos: Con herramientas como docker stats o podman stats, observen el consumo de CPU y memoria de sus contenedores durante la prueba de carga.
5. Realiza el análisis FinOps: A partir del consumo de recursos y del desempeño observado, estimen el costo por inferencia del sistema (por ejemplo, con base en un precio de referencia de una instancia en la nube) y discutan cómo optimizarlo.
6. Documenta y reporta: Integren en un reporte los resultados de la prueba de carga (con gráficas) y el análisis FinOps, y actualicen el README con las instrucciones para levantar el sistema y ejecutar el pipeline.

## Especificaciones de la Entrega 
Para considerar completa la entrega, asegúrate de cumplir con lo siguiente:

* Modalidad: Colaborativa (en equipo de 4 personas).
    * La evaluación es individual con base en el entregable del equipo y en la contribución evidenciada en el historial de Git.
* Medio de realización/entrega:
    * Enlace al repositorio de GitHub del equipo (con docker-compose.yml, el workflow de CI/CD y el código de Locust).
    * Más un reporte en PDF con las gráficas de Locust y el análisis FinOps, subidos a la plataforma Canvas.
* Formato:
    * Repositorio de GitHub funcional y reporte en PDF
        * Tipografía Arial 11 o equivalente, interlineado 1.15, de 3 a 5 cuartillas sin contar anexos.
    * El pipeline de CI/CD debe mostrarse en verde (exitoso) y el reporte debe incluir gráficas de la prueba de carga y el cálculo del costo por inferencia.
* Nombre del entregable:
    * Reporte: TC5061_Operacion_FinOps_Equipo[Número].pdf
        * Por ejemplo: TC5061_Operacion_FinOps_Equipo03.pdf.
    * Repositorio: el mismo repositorio del proyecto del equipo (TC5061-Proyecto-Equipo[Número]).

## Criterios de Evaluación 
| Criterio | Excelente (100%) | Satisfactorio (70%) | Insuficiente (<50%) | Peso |
|---|---|---|---|:---:|
Orquestación con Docker Compose: El archivo docker-compose.yml levanta correctamente los servicios del proyecto y se ejecuta con un solo comando. | El archivo docker-compose.yml configura y levanta correctamente todos los servicios especificados del proyecto. Se ejecuta exitosamente con un solo comando y está optimizado con configuraciones avanzadas (como redes, volúmenes o variables de entorno). | El archivo docker-compose.yml configura y levanta correctamente los servicios necesarios del proyecto. Se ejecuta con un solo comando, aunque con configuraciones básicas o mínimas. | El archivo docker-compose.yml levanta parcialmente los servicios del proyecto o requiere ajustes manuales para su ejecución. | 25% |
Pipeline de CI/CD con GitHub Actions: El workflow ejecuta pruebas y construye la imagen de forma automática, y se muestra exitoso. | El workflow automatiza completamente las pruebas y la construcción de la imagen del servicio, y se muestra exitoso. Incluye configuraciones adicionales que optimizan el proceso (por ejemplo, caché de dependencias, matrices de prueba, validación del levantamiento con Compose o publicación de la imagen en un registro). | El workflow automatiza las pruebas y la construcción de la imagen del servicio, mostrando un resultado exitoso, aunque sin configuraciones avanzadas. | El workflow automatiza parcialmente las pruebas o la construcción de la imagen del servicio, y puede requerir intervención manual para completarse. | 25% |
Pruebas de carga con Locust: El escenario es pertinente y el reporte incluye gráficas de desempeño interpretadas. | El escenario de pruebas es altamente pertinente para evaluar el sistema bajo carga. El reporte incluye gráficas claras y un análisis detallado que interpreta los resultados, identificando áreas de mejora. | El escenario de pruebas es pertinente y evalúa adecuadamente el sistema bajo carga. El reporte incluye gráficas y una interpretación adecuada de los resultados. | El escenario de pruebas es pertinente y evalúa adecuadamente el sistema bajo carga. El reporte incluye gráficas y una interpretación adecuada de los resultados. | El escenario de pruebas es básico o incompleto para evaluar el sistema bajo carga. El reporte incluye gráficas, pero con interpretación limitada o superficial. | 20% |
Análisis FinOps: Se estima de forma razonada el costo por inferencia y se proponen optimizaciones. | Se estima detalladamente el costo por inferencia, considerando múltiples factores (por ejemplo, tiempo, recursos y escalabilidad). Se proponen optimizaciones claras y viables, justificadas con datos. | Se estima razonadamente el costo por inferencia, considerando los factores principales. Se proponen optimizaciones generales o con justificación limitada. | La estimación del costo por inferencia es incompleta o poco razonada. No se proponen optimizaciones o estas carecen de justificación. | 20% |
Documentación y trabajo colaborativo: El README está actualizado y el historial de Git evidencia la participación de todos. | El archivo README está completamente actualizado, incluye instrucciones claras para la ejecución y detalles técnicos relevantes. El historial de Git evidencia contribuciones significativas y equilibradas de todos los integrantes del equipo. | El archivo README está actualizado y contiene las instrucciones principales para la ejecución. El historial de Git evidencia contribuciones de la mayoría de los integrantes del equipo. | El archivo README está incompleto o desactualizado, y/o el historial de Git muestra una participación limitada o desigual entre los integrantes del equipo. | 10% |
