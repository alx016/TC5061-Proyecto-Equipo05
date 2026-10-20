# MLFlow Project y Tracking Avanzado
Fecha de Entrega: Octubre 25, 2026
Integrantes: 4
## Objetivos
Esta actividad impacta con el siguiente objetivo específico de aprendizaje: 

* Implementar prácticas y técnicas que aseguren la reproducibilidad en el entrenamiento de modelos.
Adicional, esta actividad busca:

* Empaquetar el código del proyecto como un MLflow Project que cualquier integrante pueda ejecutar con un solo comando.
* Comparar sistemáticamente múltiples experimentos para P1, P2 y P3 contra la línea base registrada en la entrega anterior.
* Registrar en cada corrida los artefactos que respaldan su resultado, incluida la versión de los datos.
* Llevar el mejor modelo de cada pregunta al Model Registry, gestionando su ciclo de vida hasta el estado de producción.

Esta actividad es la segunda entrega del proyecto integrador y continúa directamente la Propuesta de proyecto de la semana anterior: se trabaja sobre el mismo repositorio (TC5061-Proyecto-EquipoNN), el mismo experimento de MLflow (mibici-equipoNN), los mismos datos versionados y las mismas tres preguntas. Si su entrega anterior quedó incompleta (ingesta, manifiesto de datos o baselines), compleméntenla primero, porque todo lo que se pide aquí se mide contra esa base.

### Qué aporta esta entrega al proyecto final
* **Un proyecto ejecutable con un solo comando:** el archivo MLproject con los puntos de entrada de ingesta y entrenamiento.
* **Evidencia de experimentación:** varias corridas por pregunta, organizadas y comparables contra la línea base.
* **Un modelo en producción por pregunta**, versionado en el Model Registry y con su linaje hasta el commit y la versión de datos que lo produjeron. Este es el modelo que consumirán las siguientes etapas del proyecto.

## Instrucciones
En esta actividad, en equipo, llevarán el uso de MLflow en el proyecto a un nivel profesional: convertirán el repositorio en un MLflow Project, ejecutarán una experimentación ordenada para P1, P2 y P3, y dejarán el mejor modelo de cada pregunta registrado y en producción.

### Punto de partida. 
Al iniciar esta actividad, su repositorio ya debe contar con lo entregado en la Propuesta de proyecto:

* Ingesta reproducible en dataset.py con su manifiesto de versión de datos.
* Funciones de feature engineering en features.py y configuración centralizada en config.py.
* Un baseline por pregunta registrado en el experimento mibici-equipoNN, con la partición temporal y las métricas que el equipo propuso.
Las buenas prácticas de MLflow, estructura y Git definidas en esa entrega siguen vigentes y se vuelven a revisar.

### Parte A. Proyecto Reproducible
1. **Configurar un servidor de tracking local con base de datos:** El Model Registry requiere un almacenamiento respaldado por base de datos. Levanten un servidor local, por ejemplo mlflow `server --backend-store-uri sqlite:///mlflow.db --port 5000`, apunten `MLFLOW_TRACKING_URI` a él y documenten el comando en el README. Agreguen `mlflow.db` al `.gitignore`.
2. **Crear el archivo `MLproject`**: En la raíz del repositorio, definan un MLflow Project con:
* Un entorno declarado (por ejemplo, python_env.yaml) con las mismas versiones fijas del archivo de dependencias.
* Al menos dos puntos de entrada: ingest, que ejecuta dataset.py, y train, que recibe como parámetros la pregunta (p1, p2 o p3), el modelo y sus hiperparámetros.
* Valores por defecto tomados de config.py, sin repetir rutas, fechas ni semillas.
3. **Ejecutar todo con `mlflow run`**: Todas las corridas de esta entrega se lanzan desde el MLflow Project (por ejemplo, `mlflow run . -e train -P question=p1 -P model=rf -P n_estimators=300`), no desde un notebook. Documenten en el README los comandos para reproducir la ingesta y el modelo elegido de cada pregunta desde un clon limpio.

### Parte B. Experimentación y Artefactos
4. **Ejecutar y comparar experimentos**: En el mismo experimento `mibici-equipoNN`, entrenen para cada pregunta al menos tres versiones adicionales al baseline, variando algoritmos, hiperparámetros o features:
* Usen al menos una búsqueda de hiperparámetros organizada como nested runs: una corrida padre por búsqueda y una corrida hija por combinación.
* Mantengan la misma partición temporal y las mismas métricas de la entrega anterior, para que la comparación contra el baseline sea válida. Si cambian alguna, justifíquenlo en el reporte.
* Agreguen etiquetas que permitan filtrar y comparar en la interfaz, por ejemplo `question`, `model_family` y `stage_of_work` (`baseline`, `search`, `candidate`), además de las de equipo y commit ya obligatorias.
5. **Registrar artefactos**: En cada corrida candidata, guarden con MLflow los artefactos que respaldan el resultado:
* El modelo con log_model, incluyendo su firma (signature) y un ejemplo de entrada.
* La versión de los datos: el manifiesto generado por la ingesta como artefacto y, de forma recomendada, el conjunto de entrenamiento con mlflow.log_input.
* Gráficas de desempeño acordes a la tarea: por ejemplo, real contra predicho o residuales (P1), matriz de confusión y curva precisión-recall (P2), y mapa o perfiles de los grupos de estaciones (P3).
* Los archivos de configuración con que se ejecutó la corrida.
Organicen los artefactos en subcarpetas (por ejemplo, `plots/`, `data/`, `config/`) y no registren datos personales de los usuarios de MiBici.
6. **Seleccionar el mejor modelo**: Con base en la comparación de métricas en la interfaz de MLflow, elijan de forma justificada el modelo de cada pregunta que pasará a producción. Consideren el desempeño respecto al baseline, la métrica ligada a la decisión de negocio de cada pregunta, la estabilidad de los resultados y el costo de entrenamiento o inferencia.

### Parte c. Model Registry y reporte
7. **Registrar el modelo en el Model Registry**: Registren el modelo elegido de cada pregunta con un nombre claro y consistente, por ejemplo `mibici-equipoNN-p1-demanda`, `mibici-equipoNN-p2-riesgo` y `mibici-equipoNN-p3-grupos`. Cada versión debe conservar su linaje: experimento, corrida de origen, commit y versión de datos. Agreguen una descripción a cada modelo registrado.
8. **Gestionar su ciclo de vida**: Lleven el modelo de cada pregunta al estado de producción:
1. Pueden hacerlo mediante el esquema clásico de etapas (transición a Production) o, preferentemente, mediante el esquema vigente de aliases (por ejemplo, el alias `champion` para la versión en producción y, de forma opcional, `challenger` para la siguiente mejor).
2. Verifiquen que el modelo en producción se puede cargar por su nombre y alias o etapa (por ejemplo, models:/`mibici-equipoNN-p1-demanda@champion`).
3. Documenten en el reporte y en el README qué esquema utilizaron y por qué.
9. **Mantener el flujo de trabajo con Git/GitHub**: Todo el trabajo entra a la rama principal protegida mediante Pull Request revisado, con las convenciones del equipo. Cada integrante abre al menos un Pull Request y revisa al menos uno en esta entrega.
10. **Elaborar el reporte de experimentos**: Redacten un reporte que incluya:
* Enlace al repositorio y comandos de `mlflow` run para reproducir los resultados.
* Una tabla comparativa por pregunta con el baseline y las corridas candidatas: modelo, hiperparámetros clave, métricas de validación y prueba, y mejora respecto al baseline.
* El análisis de la comparación y la justificación del modelo elegido por pregunta.
* El esquema de ciclo de vida utilizado y su justificación.
* Capturas de la interfaz de MLflow que muestren: la comparación de corridas con nested runs, el detalle de los artefactos de una corrida candidata y los tres modelos registrados en estado de producción con su linaje.
* Referencias en formato APA 7.
## Criterios de evaluación
La actividad se evaluará con base en la siguiente rúbrica (100 puntos en total), aplicada por igual a todos los equipos. Se evalúa la coherencia y justificación de las decisiones, no que coincidan con una solución única.

| Criterio | Excelente (100%) | Satisfactorio (70%) | Insuficiente (<50%) | Peso |
|---|---|---|---|:---:|
| **MLflow Project y reproducibilidad** | El `MLproject` declara entorno y puntos de entrada parametrizados; todas las corridas se lanzan con `mlflow run` y se reproducen desde un clon limpio siguiendo el README. No hay datos, base de datos de MLflow ni modelos en el historial. | El `MLproject` funciona, pero algún punto de entrada no está parametrizado, alguna corrida se lanzó fuera del proyecto o la reproducción requiere un paso manual. | No existe `MLproject` o no se ejecuta, o se versionaron datos o credenciales. | 15% |
| **Comparación de experimentos en MLflow Tracking** | Cada pregunta tiene al menos tres corridas candidatas comparables con su *baseline*, una búsqueda organizada con *nested runs* y etiquetas útiles. El análisis de métricas es sistemático y profundo. | Hay varias corridas por pregunta, pero alguna pregunta tiene menos de tres candidatas, faltan *nested runs* o etiquetas, o el análisis es poco detallado. | Las corridas no permiten comparar contra el *baseline*, cubren una sola pregunta o carecen de un análisis que evidencie la comparación. | 20% |
| **Registro de artefactos** | Las corridas candidatas guardan el modelo con firma, la versión de los datos, gráficas acordes a cada tarea y la configuración, organizados en subcarpetas y sin datos personales. | Se guardan artefactos relevantes en la mayoría de las corridas, con omisiones menores (por ejemplo, firma o versión de datos) o poca organización. | Los artefactos están en pocas corridas, omiten el modelo o la versión de datos, o incluyen datos personales. | 15% |
| **Registro y gestión en el Model Registry** | Los tres modelos están registrados con nombre claro, descripción y linaje completo; cada uno tiene una versión en producción cargable por nombre, y el esquema (etapas o *aliases*) se justifica con un enfoque profesional. | Los modelos están registrados y en producción, pero falta alguno, el linaje está incompleto o la justificación del esquema es superficial. | Solo una pregunta tiene modelo registrado, el nombre o la versión no son claros, o no hay versión en producción. | 20% |
| **Justificación de la selección del mejor modelo** | La elección de cada pregunta se fundamenta en la mejora sobre el *baseline*, en la métrica ligada a la decisión de negocio y en consideraciones de estabilidad y costo, con un análisis crítico. | La elección se basa en métricas y evidencia, pero la justificación es poco profunda o no se relaciona con la decisión de negocio. | La elección se basa solo en la mejor cifra de una métrica, sin evidencia sólida ni comparación con el *baseline*. | 15% |
| **Flujo de trabajo con Git/GitHub** | Todo el trabajo de esta entrega entró por *Pull Request* revisado y cada uno de los cuatro integrantes abrió al menos uno y revisó al menos uno, siguiendo las convenciones del equipo. | Hay *Pull Requests*, pero algún integrante no abrió o no revisó uno, o las convenciones se siguen de forma irregular. | El trabajo se concentra en la rama principal o en uno o dos integrantes. | 5% |
| **Claridad y completitud del reporte** | El reporte es claro, estructurado y completo, cumple el formato, incluye todas las capturas solicitadas y referencias en APA 7. El lenguaje es profesional, inclusivo y libre de ambigüedades. | El reporte está estructurado y con evidencia, pero algún apartado o captura falta o es poco claro, o hay errores menores de redacción o en las referencias. | El reporte tiene estructura incompleta, evidencia insuficiente o lenguaje ambiguo. | 10% |
