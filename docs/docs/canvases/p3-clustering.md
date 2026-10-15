# Machine Learning Canvas — P3: Clustering Funcional de Estaciones

!!! info "Ficha Técnica del Caso de Uso"
    - **Marco metodológico:** OwnML (Louis Dorard, v1.2)  
    - **Proyecto:** TC5061 - MiBici (Equipo 05)  
    - **Pregunta de negocio (P3):** ¿Qué grupos de estaciones comparten patrones de uso y dónde se ubican en la ciudad?  
    - **Tipo de tarea:** Aprendizaje no supervisado (Clustering)  
    - **Usuario principal:** Dirección de Planeación Urbana, Expansión de Red y Autoridades de Movilidad del AMG

---

## 1. Propuesta de Valor (Value Proposition)

- **Objetivo:** Descubrir y caracterizar tipologías funcionales de estaciones de MiBici agrupándolas por su dinámica temporal de uso a lo largo del día (orígenes residenciales matutinos, destinos corporativos vespertinos, corredores mixtos y nodos recreativos).
- **Valor para el usuario:** Permite fundamentar decisiones estratégicas de infraestructura urbana: priorizar ampliación de anclajes en estaciones atractivas saturadas, rediseñar ciclovías en corredores con alta demanda y planificar nuevas estaciones en zonas con perfiles complementarios.
- **Ventaja sobre agrupación geográfica simple:** Dos estaciones contiguas pueden tener funciones urbanas opuestas (una frente a oficinas y otra frente a vivienda). El clustering funcional captura el comportamiento real de movilidad de los usuarios, no solo la cercanía física.

---

## 2. Tarea Central de Machine Learning (Central ML Task)

- **Tipo de tarea:** Clustering no supervisado sobre perfiles temporales continuos.
- **Entidad / Granularidad:** Estación individual ($s \in \{1, \dots, N\}$).
- **Entrada ($X_s$):** Vector de perfil de uso normalizado que captura la distribución relativa de actividad horaria de 24 horas.
- **Salida ($Y_s$):** Asignación a un clúster funcional discreto $k \in \{1, \dots, K\}$, donde $K$ se determina empíricamente ($K \in [3, 6]$).
- **Propiedad deseada:** Cohesión interna alta (estaciones en el mismo grupo exhiben dinámicas idénticas) y separación máxima entre clusters con clara interpretabilidad urbana.

---

## 3. Decisiones de Planeación Urbana (Decisions)

- **Consumo:** Reporte analítico semestral y visualización en mapa geoespacial interactivo en el dashboard de movilidad.
- **Acciones derivadas:**
    - **Rebalanceo estructural:** Identificar estaciones con asimetría extrema para modificar permanentemente su capacidad física de anclajes.
    - **Expansión de la red:** Recomendar la instalación de nuevas estaciones en polígonos donde la demanda esté desbalanceada para actuar como contrapartes de viaje.
    - **Políticas de incentivo:** Diseñar programas de rebalanceo comunitario (incentivos tarifarios a usuarios que pedaleen desde clusters saturados hacia clusters con desabasto).

---

## 4. Inferencia y Generación de Artefactos (Making Predictions)

- **Modo:** Batch periódico (semestral / anual o bajo demanda en estudios de planeación).
- **Frecuencia:** No requiere scoring en tiempo real. Se ejecuta tras cada consolidación de periodo canónico o antes de presupuestar expansiones.
- **Latencia:** Proceso analítico offline completado en menos de 2 minutos.
- **Artefactos generados:** Mapa geográfico con código de colores por cluster y gráfica de centroides horarios promedio de 24h para cada tipología.

---

## 5. Recolección y Agregación de Datos (Data Collection)

- **Fuentes:**
    - Consolidado anual de viajes 2024 de MiBici: `Inicio_del_viaje`, `Fin_del_viaje`, `Origen_Id`, `Destino_Id`.
    - Catálogo de estaciones: latitud, longitud, nombre y capacidad.
- **Agregación:** Conteo total de salidas y llegadas por hora del día ($0\dots 23$) para cada estación a lo largo de todo el año de estudio.
- **Filtrado previo:** Exclusión de estaciones con menos de 100 viajes mensuales para evitar distorsiones por ruido estadístico.

---

## 6. Representación de Perfil de Uso (Features)

- **Perfil de salidas normalizado (24 horas):** Proporción de viajes iniciados en cada hora del día respecto al total de la estación:
    $$p^{\text{out}}_{s, h} = \frac{\sum_{\text{días}} \text{Salidas}_{s, h}}{\sum_{h=0}^{23} \sum_{\text{días}} \text{Salidas}_{s, h}}, \quad \sum_{h=0}^{23} p^{\text{out}}_{s, h} = 1$$
- **Perfil de llegadas normalizado (24 horas):** Proporción de viajes finalizados en cada hora del día:
    $$p^{\text{in}}_{s, h} = \frac{\sum_{\text{días}} \text{Llegadas}_{s, h}}{\sum_{h=0}^{23} \sum_{\text{días}} \text{Llegadas}_{s, h}}, \quad \sum_{h=0}^{23} p^{\text{in}}_{s, h} = 1$$
- **Ratio de asimetría global (Generador vs Atractor):**
    $$\text{Asimetria}_s = \frac{\text{Llegadas Totales}_s - \text{Salidas Totales}_s}{\text{Llegadas Totales}_s + \text{Salidas Totales}_s} \in [-1, 1]$$

---

## 7. Modelos y Calibración (Building Models)

- **Espacio de representación:** Concatenación del vector normalizado de 48 dimensiones ($24\text{h salidas} + 24\text{h llegadas}$) más el ratio de asimetría.
- **Algoritmos evaluados:**
    1. *K-Means:* Con inicialización K-Means++ y optimización de inercia.
    2. *Clustering Jerárquico Aglomerativo:* Con enlace Ward y distancia euclidiana.
    3. *HDBSCAN:* Para detección de agrupaciones de densidad variable y aislamiento de estaciones atípicas.
- **Selección de $K$ Óptimo:** Evaluación conjunta mediante el Método del Codo (Elbow Method), Silhouette Score medio y Davies-Bouldin Index.

---

## 8. Métricas y Validación (Evaluation)

- **Métricas No Supervisadas:**
    - **Silhouette Score:** $\ge 0.45$ (certifica separación nítida entre tipologías).
    - **Davies-Bouldin Index:** Minimización ($< 1.0$).
- **Validación Cualitativa Urbana:**
    - Los centroides deben correlacionar de manera intuitiva con los usos de suelo de Guadalajara (nodos residenciales en Zapopan/Providencia vs. nodos atractores en Zona Financiera / Chapultepec / Centro Histórico).
