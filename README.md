"""
# Proyecto de Automatización de ML en AWS

Este proyecto automatiza el entrenamiento y despliegue de un modelo de clasificación (churn) en AWS, usando SageMaker Pipelines, procesamiento con sklearn, y reentrenamiento automático vía EventBridge.

## Instrucciones Rápidas

1. Subir dataset `telco_churn.csv` a un bucket S3
2. Editar `sagemaker_pipeline.py` para que apunte al S3 correcto
3. Ejecutar el pipeline desde SageMaker Studio o CLI
4. Configurar EventBridge para reentrenar periódicamente

## Requisitos

- AWS CLI configurado
- SageMaker Studio
- IAM role con permisos para SageMaker y EventBridge

## Instalación Local

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Estructura del repositorio

- `src/`: código del modelo ML
- `pipelines/`: definición de pipeline de entrenamiento
- `infra/`: automatización y orquestación
- `notebooks/`: análisis exploratorio
- `data/`: datos de entrada

"""
