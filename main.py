from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
import sys
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import VectorAssembler
import os

def prepare_data(input_data):
    # Load wine quality dataset
    new_columns = [col.replace('"', '') for col in input_data.columns]
    input_data = input_data.toDF(*new_columns)

    label_column = 'quality'

    # 'quality' is a categorical variable, indexing it
    indexer = StringIndexer(inputCol=label_column, outputCol="label")
    input_data = indexer.fit(input_data).transform(input_data)

    # Selecting relevant feature columns
    feature_columns = [col for col in input_data.columns if col != label_column]

    # VectorAssembler to assemble feature columns into a single 'features' column
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Apply the VectorAssembler
    assembled_data = assembler.transform(input_data)

    return assembled_data

def predict_using_model(test_data_path, output_model):
    # Initialize Spark session
    spark = SparkSession.builder.appName("WineQualityPrediction").getOrCreate()

    # Load data from the local file system using the input path provided
    test_raw_data = spark.read.csv(test_data_path, header=True, inferSchema=True, sep=";")

    # Load prepared test data
    test_data = prepare_data(test_raw_data)

    # Load the trained model from the local file system
    model_path = os.path.join(os.getcwd(), output_model)
    trained_model = PipelineModel.load(model_path)

    # Make predictions
    predictions = trained_model.transform(test_data)

    # Define evaluator
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    # Evaluate the predictions
    accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
    f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})

    print(f"Test Accuracy: {accuracy}")
    print(f"Test F1 Score: {f1_score}")

    # Stop Spark session
    spark.stop()

if __name__ == "__main__":
    # Input paths for test_data, and output model as command-line arguments
    if len(sys.argv) != 3:
        #spark-submit main.py ValidationDataset.csv winemodel
        sys.exit(1)

    test_data_path = sys.argv[1]
    output_model = sys.argv[2]

    predict_using_model(test_data_path, output_model)
