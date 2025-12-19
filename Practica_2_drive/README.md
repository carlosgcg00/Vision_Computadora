# Práctica de clasificación de texturas

**Autor:** Carlos González Carballo

Este proyecto tiene como objetivo desarrollar y evaluar clasificadores de texturas utilizando técnicas de Visión por Computador y Aprendizaje Automático (ML). Se exploran diferentes descriptores de características y modelos de clasificación.

## Estructura del Proyecto

El proyecto se ha estructurado en directorios para mantener el código modular y organizado:

- **/data**: Contiene el dataset de imágenes de texturas.
- **/descriptors**: Implementación de algoritmos de extracción de características (HOG, GLCM, LBP).
- **/dataset_loader**: Funciones para la carga, preprocesamiento y división del dataset.
- **/figures**: Gráficas y visualizaciones generadas durante la ejecución.
- **/Results**: Almacenamiento de resultados de los experimentos.
- **/summary_results**: Scripts para la evaluación de métricas y generación de matrices de confusión.
- `Clasificador_texturas.ipynb`: Notebook principal que orquesta los experimentos, desde la carga de datos hasta la clasificación y evaluación.
- `config.py`: Archivo de configuración con parámetros globales (rutas, semillas aleatorias, etc.).

## Descriptores Implementados

Se han implementado y evaluado los siguientes descriptores de textura:

- **HOG (Histogram of Oriented Gradients)**: Captura la distribución de las direcciones del gradiente.
- **GLCM (Gray-Level Co-occurrence Matrix)**: Analiza la textura basándose en la relación espacial de los niveles de gris (contraste, disimilitud, homogeneidad, energía, correlación).
- **LBP (Local Binary Patterns)**: Codifica la textura local comparando cada píxel con sus vecinos.

## Requisitos

El proyecto requiere Python y las siguientes librerías principales:

- `numpy`
- `matplotlib`
- `opencv-python` (cv2)
- `scikit-learn`
- `scikit-image`
- `jupyter` (para ejecutar el notebook)

Puedes instalar las dependencias usando `pip`:

```bash
pip install numpy matplotlib opencv-python scikit-learn scikit-image jupyter
```

## Uso

1.  Asegúrate de tener los datos en la carpeta `data`.
2.  Configura los parámetros en `config.py` si es necesario.
3.  Abre y ejecuta el notebook `Clasificador_texturas.ipynb` para reproducir los experimentos y entrenar los modelos.

```bash
jupyter notebook Clasificador_texturas.ipynb
```
