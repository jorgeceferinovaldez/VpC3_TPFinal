<img src="https://github.com/hernancontigiani/ceia_memorias_especializacion/raw/master/Figures/logoFIUBA.jpg" width="500" align="center">

<hr>

# Maestría en Inteligencia Artificial

# Visión por Computadora 3

# Trabajo Práctico Número 3

---
# Integrantes:
   - **Jorge Ceferino Valdez**
   - **Matías Marando**
   - **Fabian Sarmiento**

# Detección de Anomalías en MVTec AD con Swin Transformer

Este proyecto implementa un sistema de detección de anomalías utilizando Swin Transformer en el dataset MVTec AD. Se implementan dos enfoques diferentes: supervisado y no supervisado (autoencoder).

## 📋 Descripción

El proyecto utiliza Vision Transformers (específicamente Swin Transformer) para detectar anomalías en imágenes industriales del dataset MVTec AD. Se incluyen dos aproximaciones:

- **Modelo Supervisado**: Clasificación binaria normal vs. anómalo
- **Modelo Autoencoder**: Detección de anomalías basada en error de reconstrucción

## 🏗️ Estructura del Proyecto

```
├── data/                          # Dataset MVTec AD (se descarga automáticamente)
├── models/                        # Modelos entrenados guardados
├── reports/                       # Reportes, gráficos y visualizaciones
│   ├── examples/                  # Ejemplos de predicciones
│   ├── anomaly_maps/             # Mapas de anomalías
│   ├── curvas_de_evaluacion.png  # Curvas ROC y Precision-Recall
│   ├── confusion_matrix.png      # Matriz de confusión
│   └── resultados_finales.json   # Métricas finales
├── 1.0.download-dataset.ipynb    # Notebook: Descarga y extrae el dataset
├── 2.0-EDA.ipynb                 # Notebook: Análisis exploratorio de datos
├── 3.0.model-training.ipynb      # Notebook: Entrenamiento y evaluación
├── requirements.txt              # Dependencias del proyecto
└── README.md                     # Este archivo
```

## 🚀 Instalación

### 1. Clonar el repositorio
```bash
git clone https://github.com/jorgeceferinovaldez/VpC3_TPFinal.git
cd mvtec-anomaly-detection
```

### 2. Crear entorno virtual (recomendado)
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 3. Instalar dependencias
```bash
pip install -r requirements.txt
```

## 📊 Dataset

El proyecto utiliza el dataset **MVTec AD** (Anomaly Detection), que contiene:
- 15 categorías de objetos industriales
- Imágenes normales para entrenamiento
- Imágenes normales y anómalas para test
- Máscaras de ground truth para anomalías

### Categorías incluidas:
- bottle, cable, capsule, carpet, grid
- hazelnut, leather, metal_nut, pill, screw
- tile, toothbrush, transistor, wood, zipper

## 🎯 Uso

### 1. Descargar y preparar el dataset
Ejecuta el notebook `1.0.download-dataset.ipynb` para descargar automáticamente el dataset MVTec AD desde Google Drive y extraerlo en la carpeta `data/`.

### 2. Análisis exploratorio de datos (opcional)
Ejecuta el notebook `2.0-EDA.ipynb` para:
- Visualizar la distribución de imágenes por categoría
- Explorar ejemplos de imágenes normales y anómalas
- Generar gráficos estadísticos del dataset

### 3. Entrenamiento y evaluación
Ejecuta el notebook `3.0.model-training.ipynb` para:
- Entrenar el modelo seleccionado (supervisado o autoencoder)
- Evaluar el rendimiento en el conjunto de test
- Generar visualizaciones y reportes

## ⚙️ Configuración

En el notebook `3.0.model-training.ipynb` puedes modificar las siguientes variables en las celdas correspondientes:

```python
# Tipo de modelo
model_type = 'autoencoder'  # o 'supervisado'

# Hiperparámetros
batch_size = 8
lr = 1e-4
num_epochs = 10  # Ajustar según necesidades
pretrained = True
```

### Ejecución en Jupyter

Para ejecutar los notebooks:

1. **Iniciar Jupyter Lab/Notebook**:
```bash
jupyter lab
# o
jupyter notebook
```

2. **Ejecutar notebooks en orden**:
   - `1.0.download-dataset.ipynb` (una sola vez)
   - `2.0-EDA.ipynb` (opcional, para exploración)
   - `3.0.model-training.ipynb` (entrenamiento principal)

## 🏛️ Arquitectura de los Modelos

### Modelo Supervisado
- **Backbone**: Swin Transformer Tiny pre-entrenado
- **Cabeza**: Red neuronal con capas lineales, BatchNorm, ReLU y Dropout
- **Salida**: Probabilidad de anomalía (0-1)
- **Pérdida**: Binary Cross Entropy (BCE)

### Modelo Autoencoder
- **Encoder**: Swin Transformer Tiny pre-entrenado
- **Decoder**: Red neuronal que reconstruye la imagen
- **Detección**: Error de reconstrucción MSE
- **Pérdida**: Mean Squared Error (MSE)

## 📈 Métricas de Evaluación

El sistema evalúa los modelos utilizando:
- **ROC AUC**: Área bajo la curva ROC
- **Precision-Recall AUC**: Área bajo la curva Precision-Recall
- **Accuracy**: Precisión general
- **Precision, Recall, F1-Score**: Métricas de clasificación binaria

## 📊 Visualizaciones

El proyecto genera automáticamente:

1. **Análisis Exploratorio**:
   - Distribución de imágenes por categoría
   - Ejemplos de imágenes normales y anómalas

2. **Resultados de Evaluación**:
   - Curvas ROC y Precision-Recall
   - Matriz de confusión
   - Ejemplos de predicciones (TP, TN, FP, FN)
   - Mapas de anomalías (para autoencoder)

## 🔧 Características Técnicas

- **Framework**: PyTorch
- **Transformaciones**: Redimensionado, normalización, data augmentation
- **Optimizador**: AdamW con weight decay
- **Scheduler**: Cosine Annealing Learning Rate
- **Dispositivo**: Soporte automático para CUDA, MPS (Apple Silicon) y CPU
- **Reproducibilidad**: Seeds fijas para resultados consistentes

## 📋 Resultados Esperados

El modelo genera los siguientes archivos de salida:

```
models/
├── mejor_modelo_supervisado.pth      # Mejor modelo supervisado
└── mejor_modelo_multiclase_autoencoder.pth  # Mejor modelo autoencoder

reports/
├── curvas_de_evaluacion.png         # Curvas ROC y PR
├── confusion_matrix.png             # Matriz de confusión
├── resultados_finales.json          # Métricas en formato JSON
├── examples/                        # Ejemplos de predicciones
│   ├── true_positive.png
│   ├── true_negative.png
│   ├── false_positive.png
│   └── false_negative.png
└── anomaly_maps/                    # Mapas de anomalías
    ├── normal_0.png
    ├── normal_1.png
    ├── anomaly_0.png
    └── anomaly_1.png
```

## 🛠️ Requisitos del Sistema

- Python 3.8+
- 8GB+ RAM recomendado
- GPU con CUDA (opcional, mejora significativamente el rendimiento)
- ~5GB de espacio en disco para el dataset

## 🐛 Solución de Problemas

### Error de memoria
```python
# Reducir batch_size en 3.0.model-training.py
batch_size = 4  # En lugar de 8
```

### Error de CUDA
```python
# El código detecta automáticamente el dispositivo disponible
# Si hay problemas con CUDA, forzar CPU:
device = "cpu"
```

### Dataset no encontrado
```bash
# Ejecutar nuevamente el notebook de descarga
# Abrir y ejecutar: 1.0.download-dataset.ipynb
```

### Error al ejecutar notebooks
- Asegúrate de que Jupyter esté instalado: `pip install jupyter`
- Inicia Jupyter Lab: `jupyter lab`
- Ejecuta las celdas en orden secuencial

## 📚 Referencias

- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
- [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- [Timm: PyTorch Image Models](https://github.com/rwightman/pytorch-image-models)

## 📄 Licencia

Este proyecto está bajo la licencia MIT. Ver el archivo LICENSE para más detalles.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Por favor, abre un issue para discutir cambios mayores antes de enviar un pull request.

## 📧 Contacto

Para preguntas o problemas, por favor abre un issue en el repositorio.

---

**Nota**: Este proyecto es para fines educativos y de investigación. Los resultados pueden variar según el hardware y la configuración utilizada.