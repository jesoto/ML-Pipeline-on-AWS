import sagemaker
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.parameters import ParameterString
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.sklearn.estimator import SKLearn

# Parámetros del pipeline
input_data = ParameterString(name="InputData", default_value="s3://ml-automatizacion-demo/raw/telco_churn.csv")

# Procesamiento
sklearn_processor = SKLearnProcessor(framework_version="0.23-1",
                                     role="SageMakerExecutionRole",
                                     instance_type="ml.m5.xlarge",
                                     instance_count=1)

processing_step = ProcessingStep(
    name="PreprocessData",
    processor=sklearn_processor,
    code="src/preprocessing.py",
    inputs=[],
    outputs=[],
)

# Entrenamiento
sklearn_estimator = SKLearn(entry_point="src/train.py",
                            role="SageMakerExecutionRole",
                            instance_type="ml.m5.xlarge",
                            framework_version="0.23-1")

training_step = TrainingStep(
    name="TrainModel",
    estimator=sklearn_estimator,
    inputs={"train": processing_step.properties.ProcessingOutputConfig.Outputs[0].S3Output.S3Uri},
)

pipeline = Pipeline(
    name="MLAutomatizadoPipeline",
    parameters=[input_data],
    steps=[processing_step, training_step]
)
