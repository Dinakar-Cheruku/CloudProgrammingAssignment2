from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
import os
import sys


def prepare_data(input_data):
    # Clean column names
    new_columns = [col.replace('"', '') for col in input_data.columns]
    input_data = input_data.toDF(*new_columns)

    label_column = 'quality'

    # Index the label column
    indexer = StringIndexer(inputCol=label_column, outputCol="label")
    input_data = indexer.fit(input_data).transform(input_data)

    # Assemble feature columns
    feature_columns = [col for col in input_data.columns if col != label_column]
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
    prepared_data = assembler.transform(input_data)

    return prepared_data


def train_model(train_data_path, validation_data_path, output_model):
    # Initialize a Spark session
    spark = SparkSession.builder.appName("WineQualityTraining").getOrCreate()

    # Load training and validation datasets
    train_raw_data = spark.read.csv(train_data_path, header=True, inferSchema=True, sep=";")
    validation_raw_data = spark.read.csv(validation_data_path, header=True, inferSchema=True, sep=";")

    # Prepare datasets
    train_data = prepare_data(train_raw_data)
    validation_data = prepare_data(validation_raw_data)

    # Define the classifier
    rf = RandomForestClassifier(labelCol="label", featuresCol="features")

    # Define parameter grid for hyperparameter tuning
    param_grid = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20, 30]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .build()

    # Define evaluator
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label", predictionCol="prediction", metricName="f1"
    )

    # Cross-validation for model selection
    cross_validator = CrossValidator(
        estimator=rf,
        estimatorParamMaps=param_grid,
        evaluator=evaluator,
        numFolds=3
    )

    # Train the model
    cv_model = cross_validator.fit(train_data)

    # Evaluate on validation set
    predictions = cv_model.transform(validation_data)
    f1_score = evaluator.evaluate(predictions)

    print(f"Validation F1 Score: {f1_score}")

    # Save the best model
    best_model = cv_model.bestModel
    model_path = os.path.join(os.getcwd(), output_model)
    best_model.write().overwrite().save(model_path)

    # Stop Spark session
    spark.stop()


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: spark-submit train.py <train_data_path> <validation_data_path> <output_model>")
        sys.exit(1)

    train_data_path = sys.argv[1]
    validation_data_path = sys.argv[2]
    output_model = sys.argv[3]

    train_model(train_data_path, validation_data_path, output_model)

