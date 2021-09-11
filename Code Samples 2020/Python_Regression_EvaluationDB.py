# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark Evaluating Regression in Databricks
# MAGIC 
# MAGIC ### Jongwook Woo (jwoo5@calstatela.edu), revised on 04/11/2020, 04/16/2018
# MAGIC Tested in Python 2.4.5 with Scala 2.11, Python 2 with Spark 2.1

# COMMAND ----------

# MAGIC %md ## Evaluating a Regression Model
# MAGIC 
# MAGIC In this exercise, you will create a pipeline for a linear regression model, and then test and evaluate the model.
# MAGIC 
# MAGIC ### Prepare the Data
# MAGIC 
# MAGIC First, import the libraries you will need and prepare the training and test data:

# COMMAND ----------

# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession


# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO 0: Run the code in PySpark CLI
# MAGIC 1. Set the following to True:
# MAGIC ```
# MAGIC PYSPARK_CLI = True
# MAGIC ```
# MAGIC 1. You need to generate py (Python) file: File > Export > Source File
# MAGIC 1. Run it at your Hadoop/Spark cluster:
# MAGIC ```
# MAGIC $ spark-submit Python_Regression_Evaluation.py
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

# MAGIC %md ### Load flights table

# COMMAND ----------

# Load the source data
# csv = spark.read.csv('wasb:///data/flights.csv', inferSchema=True, header=True)
if PYSPARK_CLI:
    csv = spark.read.csv('flights.csv', inferSchema=True, header=True)
else:
    csv = spark.sql("SELECT * FROM flights_csv")


# COMMAND ----------


# Select features and label
data = csv.select("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay", col("ArrDelay").alias("label"))

# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")

# COMMAND ----------

# MAGIC %md ### Define the Pipeline and Train the Model
# MAGIC Now define a pipeline that creates a feature vector and trains a regression model

# COMMAND ----------

# Define the pipeline
assembler = VectorAssembler(inputCols = ["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"], outputCol="features")
lr = LinearRegression(labelCol="label",featuresCol="features", maxIter=10, regParam=0.3)
pipeline = Pipeline(stages=[assembler, lr])

# Train the model
piplineModel = pipeline.fit(train)

# COMMAND ----------

# MAGIC %md ### Test the Model
# MAGIC Now you're ready to apply the model to the test data.

# COMMAND ----------

prediction = piplineModel.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show()

# COMMAND ----------

# MAGIC %md ### Examine the Predicted and Actual Values
# MAGIC You can plot the predicted values against the actual values to see how accurately the model has predicted. In a perfect model, the resulting scatter plot should form a perfect diagonal line with each predicted value being identical to the actual value - in practice, some variance is to be expected.
# MAGIC Run the cells below to create a temporary table from the **predicted** DataFrame and then retrieve the predicted and actual label values using SQL. You can then display the results as a scatter plot, specifying **-** as the function to show the unaggregated values.

# COMMAND ----------

predicted.createOrReplaceTempView("regressionPredictions")

# COMMAND ----------

# MAGIC %md
# MAGIC ### data visualization using SQL in Databricks, 

# COMMAND ----------

# MAGIC %md
# MAGIC ### TODO 1: Visualize the following sql as scatter plot. 
# MAGIC 1. Then, select the icon graph "Show in Dashboard Menu" in the right top of the cell to create a Dashboard
# MAGIC 1. Select "+Add to New Dashboard" and will move to new web page with the scatter plot chart
# MAGIC 1. Name the dashboard to __Regression Evaluation__
# MAGIC 
# MAGIC __NOTEL__: _%sql_ does not work at PySpark CLI but only at Databricks notebook.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT trueLabel, prediction FROM regressionPredictions

# COMMAND ----------

# MAGIC %md ### Retrieve the Root Mean Square Error (RMSE)
# MAGIC There are a number of metrics used to measure the variance between predicted and actual values. Of these, the root mean square error (RMSE) is a commonly used value that is measured in the same units as the predicted and actual values - so in this case, the RMSE indicates the average number of minutes between predicted and actual flight delay values. You can use the **RegressionEvaluator** class to retrieve the RMSE.

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print ("Root Mean Square Error (RMSE):", rmse)

# COMMAND ----------

# MAGIC %md ### Result shows:
# MAGIC #### Root Mean Square Error (RMSE): 13.1998243722
# MAGIC 
# MAGIC The result is how on average - how many minutes - are in this spark prediction down to.
# MAGIC On average we're work **13 minutes** or so on.
# MAGIC where most of our guesses are kind within 30 minutes but that's the
# MAGIC average distance of which were wrong, when we make a prediction. 
# MAGIC 
# MAGIC #### In the next topic - Parameter Tuning  - how we can improve and the performance of our model by playing with some other parameters

# COMMAND ----------


