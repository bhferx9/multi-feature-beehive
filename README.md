# Multi-feature Fusion for Automatic Classification of Beehive Sound Activity  
**Asesor:** Dr. Fernando Wario Vázquez  

## 🎯 Objetivo General  
Implementar una solución integral que permita clasificar de manera confiable el sonido de una colmena de abejas durante la actividad de pecoreo, agrupando los segmentos de audio según la distancia a la fuente de comida. El trabajo toma como referencia el estudio de [Akyol et al., 2023](https://www.sciencedirect.com/science/article/abs/pii/S0010482523002330).

## 🎯 Objetivos Específicos  
- 📚 **Revisión bibliográfica:** Analizar los artículos disponibles en la carpeta `Papers` y ampliar la revisión conforme se desarrollen nuevas necesidades.  
- ⚙️ **Implementación de algoritmos base:** Desarrollar versiones básicas de los algoritmos PSO (Particle Swarm Optimization) y CCO (Cognitive Cooperative Optimization) usando funciones *benchmark*.  
- 🔧 **Adaptación de algoritmos:** Modificar los algoritmos PSO y CCO para la tarea de reducción de características en el contexto de clasificación de audio.  
- 🧠 **Clasificación:** Utilizar clasificadores SVM y KNN para distinguir entre sonidos de colmenas según características extraídas de las señales.

## 🧪 Metodología  
1. Implementación de algoritmos base (PSO y CCO) y evaluación con funciones *benchmark*.  
2. Extracción de características acústicas de los audios (MFCC, espectrogramas, energía, etc.).  
3. Adaptación de PSO y CCO para seleccionar subconjuntos óptimos de características.  
4. Clasificación usando modelos SVM y KNN.  
5. Comparación de resultados con y sin reducción de características.

🔗 Repositorio de audios: [Zenodo - Beehive Sounds Dataset](https://zenodo.org/records/12790080)

## ✅ Resultados Esperados  
- Scripts funcionales en Python con implementación de PSO, CCO, SVM y KNN.  
- Tablas comparativas de métricas de clasificación (precisión, recall, F1-score).  
- Conclusiones detalladas sobre el rendimiento de los algoritmos con y sin reducción de características.

## 📦 Entregables  
- 🧮 Implementación de funciones base de PSO y CCO en Python, para diferentes dimensiones y escalas.  
- 📓 Notebooks en Jupyter con todo el procesamiento, análisis y resultados (subidos en este repositorio).  
- 📄 Reporte técnico redactado en Overleaf (con enlace al documento).  
- 🗣️ Presentación oral de resultados finales del proyecto.
