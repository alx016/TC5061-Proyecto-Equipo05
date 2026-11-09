# Entregable 4

## Datos generales
Fecha de entrega: Domingo 8 de Noviembre.

## Objetivo
* Aplicar distintas estrategias de despliegue de modelos de ML.
* Empaquetar el modelo del proyecto integrador MiBici del equipo como una API REST construida con FastAPI y desplegarla dentro de un contenedor gestionado con Podman o Docker (según prefiera el equipo), de modo que el modelo pueda recibir peticiones y devolver predicciones de forma reproducible en cualquier entorno.
Esta actividad corresponde a una fase de avance del proyecto MiBici: a partir del modelo registrado en el Model Registry, el equipo construye la primera versión desplegable del sistema, sentando la base sobre la que en la siguiente semana se montará la arquitectura de microservicios y la automatización.

Seguiremos trabajando con el proyecto MiBici, construido a partir de los datos abiertos de MiBici: esta entrega se construye sobre el mismo repositorio del equipo y conserva el trabajo de las entregas anteriores (modelo registrado, pruebas y chequeos de seguridad).

**Motor de contenedores**: el equipo puede trabajar con Podman o con Docker, según prefiera. Ambos siguen el estándar OCI, por lo que el mismo Containerfile (o Dockerfile) funciona con cualquiera de los dos.

## Instrucciones
En esta actividad integradora, como equipo darán el primer paso del despliegue: convertir el modelo del proyecto MiBici en un servicio consumible. El objetivo es que el modelo deje de vivir solo en un notebook y quede accesible mediante peticiones HTTP, empaquetado en un contenedor que funcione igual en cualquier máquina.

Realicen en equipo los siguientes pasos:

1. **Recupera el modelo:** Carguen al menos uno de los modelos del proyecto MiBici (regresión, clasificación o clustering) desde el Model Registry de MLFlow, utilizando su nombre y versión (o alias). Dentro del contenedor, la ubicación del registry se configura con la variable de entorno MLFLOW_TRACKING_URI (por ejemplo, http://host.containers.internal:5000 cuando el servidor de MLFlow corre en su máquina), o bien empaquetando en la imagen el modelo descargado del registry; documenten en el README la opción elegida.
2. **Construye la API con FastAPI**: Creen una API REST con al menos un endpoint de predicción que reciba datos de entrada con las variables que utiliza su modelo de MiBici, ejecute el modelo y devuelva la predicción en formato JSON.
3. **Valida las entradas**: Definan el esquema de los datos de entrada y salida (por ejemplo, con modelos de Pydantic) para asegurar que la API reciba datos correctos.
4. **Escribe el Containerfile**: Creen un Containerfile (compatible con el estándar OCI; también se acepta el nombre Dockerfile) que empaquete la aplicación con todas sus dependencias, de modo que la API pueda ejecutarse dentro de un contenedor.
5. **Construye y prueba el contenedor**: Generen la imagen con podman build (o docker build), ejecuten el contenedor localmente con podman run (o docker run) y comprueben que el endpoint responde correctamente a una petición de prueba.
6. **Documenta el servicio**: Actualicen el README con las instrucciones para construir la imagen, levantar el contenedor y consumir la API con el motor de contenedores elegido (Podman o Docker), incluyendo un ejemplo de petición y respuesta, y justifiquen brevemente la estrategia de despliegue elegida para el modelo (por ejemplo, inferencia en tiempo real o por lotes).

## Criterios de Evaluación
La actividad se evaluará con base en la siguiente rúbrica (100 puntos en total):

| Criterio | Excelente (100%) | Satisfactorio (70%) | Insuficiente (<50%) | Peso |
|---|---|---|---|:---:|
Recuperación del modelo desde el Model Registry y su correcta integración en el servicio. | Recupera el modelo desde el Model Registry utilizando el nombre y versión correctos. Integra el modelo en la API asegurando un funcionamiento óptimo y sin errores.| Recupera el modelo desde el Model Registry utilizando el nombre y versión correctos. Integra el modelo en la API, aunque presenta pequeños errores no críticos en su funcionamiento. | Recupera el modelo desde el Model Registry, pero la integración en la API tiene errores significativos que afectan el funcionamiento del servicio. | 20% |
Construcción de la API REST con FastAPI (endpoint de predicción funcional y validación de entradas/salidas). | Construye una API REST funcional con FastAPI que incluye un endpoint de predicción completamente operativo. Implementa validaciones robustas para entradas y salidas, asegurando que los datos sean correctos. | Construye una API REST funcional con FastAPI que incluye un endpoint de predicción operativo. Implementa validaciones para entradas y salidas, pero estas son parciales o presentan carencias menores. | Construye una API REST con FastAPI, pero el endpoint de predicción presenta errores graves o no es funcional. Las validaciones de entradas y salidas son insuficientes o están ausentes. | 30% |
Contenerización con Podman o Docker (Containerfile correcto e imagen que se construye y ejecuta sin errores). | Diseña un Containerfile completamente funcional que empaqueta todas las dependencias necesarias. Genera y ejecuta la imagen correctamente con Podman o Docker, comprobando que el contenedor responde a las peticiones de manera reproducible en cualquier entorno. | Diseña un Containerfile funcional que empaqueta las dependencias necesarias. Genera y ejecuta la imagen con Podman o Docker, aunque con errores menores en la ejecución del contenedor o en su reproducibilidad. | Diseña un Containerfile con errores graves que impiden la construcción de la imagen, o bien, la imagen no se ejecuta correctamente en el contenedor. | 30% |
Documentación del servicio (README con instrucciones y ejemplo de petición y respuesta). | Crea un README claro, completo y bien estructurado que incluye instrucciones detalladas para construir la imagen, levantar el contenedor y consumir la API. Proporciona un ejemplo funcional de petición y respuesta. | Crea un README que incluye las instrucciones para construir la imagen, levantar el contenedor y consumir la API, pero omite detalles menores o presenta ejemplos incompletos. | Crea un README que incluye instrucciones incompletas o confusas, y no proporciona un ejemplo funcional de petición y respuesta. | 10% |
Trabajo colaborativo evidenciado en el historial de Git. | Evidencia un trabajo colaborativo activo, consistente y equitativo en el historial de Git, con contribuciones claras de todos los integrantes del equipo. | Evidencia un trabajo colaborativo en el historial de Git, aunque con una distribución desigual de las contribuciones o con participación limitada de algunos integrantes. | Evidencia un trabajo colaborativo insuficiente en el historial de Git, con contribuciones mínimas o desbalanceadas entre los integrantes del equipo. | 10% |
