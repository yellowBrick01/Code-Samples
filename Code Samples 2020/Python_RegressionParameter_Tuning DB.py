# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark Regression with Parameters Tunning in Databricks
# MAGIC 
# MAGIC ### Jongwook Woo (jwoo5@calstatela.edu), revised on 04/11/2020, 04/16/2018
# MAGIC Tested in Python 2.4.5 with Scala 2.11, Python 2 with Spark 2.1

# COMMAND ----------

# MAGIC %md ## Tuning Model Parameters
# MAGIC 
# MAGIC In this exercise, you will optimise the parameters for a classification model.
# MAGIC 
# MAGIC ### Prepare the Data
# MAGIC 
# MAGIC First, import the libraries you will need and prepare the training and test data:

# COMMAND ----------

# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


# COMMAND ----------

# MAGIC %md ### TODO 0: Run the code in PySpark CLI
# MAGIC 1. Set the following to True:
# MAGIC ```
# MAGIC PYSPARK_CLI = True
# MAGIC ```
# MAGIC 1. You need to generate py (Python) file: File > Export > Source File
# MAGIC 1. Run it at your Hadoop/Spark cluster:
# MAGIC ```
# MAGIC $ spark-submit Python_Regression_Parameter_Tuning.py
# MAGIC ```

# COMMAND ----------

PYSPARK_CLI = True
if PYSPARK_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# DataFrame Schema, that should be a Table schema by Jongwook Woo (jwoo5@calstatela.edu) 01/07/2016
flightSchema = StructType([
  StructField("DayofMonth", IntegerType(), False),
  StructField("DayOfWeek", IntegerType(), False),
  StructField("Carrier", StringType(), False),
  StructField("OriginAirportID", IntegerType(), False),
  StructField("DestAirportID", IntegerType(), False),
  StructField("DepDelay", IntegerType(), False),
  StructField("ArrDelay", IntegerType(), False),
])

# COMMAND ----------

# MAGIC %md ### Load Flights table

# COMMAND ----------

if PYSPARK_CLI:
    csv = spark.read.csv('flights.csv', inferSchema=True, header=True)
else:
    csv = spark.sql("SELECT * FROM flights_csv")

# Load the source data
# csv = spark.read.csv('wasb:///data/flights.csv', inferSchema=True, header=True)

# COMMAND ----------

# Select features and label
# Logistic Regression
# data = csv.select("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay", ((col("ArrDelay") > 15).cast("Int").alias("label")))

# Linear Regression
data = csv.select("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay", col("ArrDelay").alias("label"))

# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")

# COMMAND ----------

# MAGIC %md ### Define the Pipeline
# MAGIC Now define a pipeline that creates a feature vector and trains a classification model

# COMMAND ----------

# Define the pipeline
assembler = VectorAssembler(inputCols = ["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"], outputCol="features")
#lr = LogisticRegression(labelCol="label", featuresCol="features")
lr = LinearRegression(labelCol="label", featuresCol="features")
pipeline = Pipeline(stages=[assembler, lr])

# COMMAND ----------

# MAGIC %md ### Tune Parameters
# MAGIC You can tune parameters to find the best model for your data. A simple way to do this is to use  **TrainValidationSplit** to evaluate each combination of parameters defined in a **ParameterGrid** against a subset of the training data in order to find the best performing parameters.
# MAGIC 
# MAGIC #### Regularization 
# MAGIC is a way of avoiding Imbalances in the way that the data is trained against the training data so that the model ends up being over fit to the training data. In other words It works really well with the training data but it doesn't generalize well with other data.
# MAGIC That we can use a **regularization parameter** to vary the way that the model balances that way.
# MAGIC 
# MAGIC #### Training ratio of 0.8
# MAGIC it's going to use 80% of the the data that it's got in its training set to train the model and then the remaining 20% is going to use to validate the trained model. 
# MAGIC 
# MAGIC In **ParamGridBuilder**, all possible combinations are generated from regParam, maxIter, threshold. So it is going to try each combination of the parameters with 80% of the the data to train the model and 20% to to validate it.

# COMMAND ----------

# LogisticRegression with attribute 'threshold' in ParamGridBuilder and BinaryClassificationEvaluator
# paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.1, 0.01]).addGrid(lr.maxIter, [10, 5]).addGrid(lr.threshold, [0.35, 0.30]).build()
# tvs = TrainValidationSplit(estimator=pipeline, evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)

# 'LinearRegression' object with RegressionEvaluator has no attribute 'threshold' in ParamGridBuilder
paramGrid = ParamGridBuilder().addGrid(lr.regParam, [0.3, 0.1, 0.01]).addGrid(lr.maxIter, [10, 5]).build()
tvs = TrainValidationSplit(estimator=pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
model = tvs.fit(train)

# COMMAND ----------

# MAGIC %md ### Test the Model
# MAGIC Now you're ready to apply the model to the test data.

# COMMAND ----------

prediction = model.transform(test)
# LogisticRegression
#predicted = prediction.select("features", "prediction", "probability", "trueLabel")

# LinearRegression
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(100)

# COMMAND ----------

# MAGIC %md ### Compute Confusion Matrix Metrics: Only for Classification Logistic Regression not for Linear Regression
# MAGIC Classifiers are typically evaluated by creating a *confusion matrix*, which indicates the number of:
# MAGIC - True Positives
# MAGIC - True Negatives
# MAGIC - False Positives
# MAGIC - False Negatives
# MAGIC 
# MAGIC From these core measures, other evaluation metrics such as *precision* and *recall* can be calculated.
# MAGIC 
# MAGIC ### Result
# MAGIC Precision (0.8762570727816253), Recall (0.7303376371612134): Precision becomes a little bit lower but the precision becomes much higher than previous no tuning example.

# COMMAND ----------

# Only for Classification Logistic Regression not for Linear Regression
'''
tp = float(predicted.filter("prediction == 1.0 AND truelabel == 1").count())
fp = float(predicted.filter("prediction == 1.0 AND truelabel == 0").count())
tn = float(predicted.filter("prediction == 0.0 AND truelabel == 0").count())
fn = float(predicted.filter("prediction == 0.0 AND truelabel == 1").count())
metrics = spark.createDataFrame([
 ("TP", tp),
 ("FP", fp),
 ("TN", tn),
 ("FN", fn),
 ("Precision", tp / (tp + fp)),
 ("Recall", tp / (tp + fn))],["metric", "value"])
metrics.show()

'''

# COMMAND ----------

predicted.createOrReplaceTempView("regressionPredictions")

# COMMAND ----------

# MAGIC %md ### data visualization using SQL in Databricks, 
# MAGIC  
# MAGIC #### TODO 1: Visualize the following sql as scatter plot. 
# MAGIC 1. Then, select the icon graph "Show in Dashboard Menu" in the right top of the cell to create a Dashboard
# MAGIC 1. Select "+Add to New Dashboard" and will move to new web page with the scatter plot chart
# MAGIC 1. Name the dashboard to __Regression Parameter Tunning__
# MAGIC 
# MAGIC __NOTEL__: _%sql_ does not work at PySpark CLI but only at Databricks notebook.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT trueLabel, prediction FROM regressionPredictions

# COMMAND ----------

# MAGIC %md ### Review the Area Under ROC: Only for Classification Logistic Regression not for Linear Regression
# MAGIC Another way to assess the performance of a classification model is to measure the area under a ROC curve for the model. the spark.ml library includes a **BinaryClassificationEvaluator** class that you can use to compute this.

# COMMAND ----------

# LogisticRegression: rawPredictionCol="prediction", metricName="areaUnderROC"
'''
evaluator = BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC")
aur = evaluator.evaluate(prediction)
print "AUR = ", aur
'''
# LinearRegression: predictionCol="prediction", metricName="rmse"
evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print("Root Mean Square Error (RMSE):", rmse)


# COMMAND ----------


