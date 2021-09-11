# Databricks notebook source
# MAGIC %md ## Midterm 2 of CIS5560 (50%)
# MAGIC ## Classification: Tuning Model Parameters
# MAGIC ##### Jongwook Woo (jwoo5@calstatela.edu)
# MAGIC 
# MAGIC In this exercise, you will optimise the parameters for a classification model using 2 **TrainValidationSplit** and 1 **CrossValidator**.
# MAGIC There are total 7 questions (50%) that you have to complete **[Fill-In]** under TODO comment of each question in the below. 
# MAGIC 
# MAGIC __NOTE__: 
# MAGIC 1. You need to create a cluster to run this code: (Default Apache Spark you have created at the labs)
# MAGIC 1. You should not change any code except [Fill-In]. If you do, you will get very low score. 
# MAGIC 1. It may take about **15 minutes** to run all cells after you complete your code except Questions 6 and 7.
# MAGIC 
# MAGIC ####Submission
# MAGIC 
# MAGIC 1. You need to submit an html file generated from this jupyter at Databricks: File > Export > HTML
# MAGIC 1. Take a screenshot of the result from __spark-submit__ with your _python_ code, which shows your account information at CLI.

# COMMAND ----------

# MAGIC %md ### Prepare the Data
# MAGIC 
# MAGIC First, import the libraries you will need and prepare the training and test data:

# COMMAND ----------

# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier
from pyspark.context import *
from pyspark.sql.session import SparkSession


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

# MAGIC %md ## Question 1 (5%)
# MAGIC 
# MAGIC Read your flights.csv file from its table at Databricks

# COMMAND ----------

# TODO: use flightSchema in order to adopt shcmea to read csv data set in the schema. 
# Jongwook Woo (jwoo5@calstatela.edu) 01/07/2016 

# Load the source data
if PYSPARK_CLI:
    csv = spark.read.csv('flights.csv', inferSchema=True, header=True)
else:
    csv = spark.sql("SELECT * FROM flights_csv")

# COMMAND ----------

# Select features and label
data = csv.select("DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay", ((col("ArrDelay") > 15).cast("Double").alias("label")))

# Split the data
splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")

# COMMAND ----------

# MAGIC %md ### Define the Pipeline
# MAGIC Now define a pipeline that creates a feature vector and trains a classification model

# COMMAND ----------

# Define the pipeline
lr = []
pipeline = []
assembler = []
for i in range(3):
  assembler.insert(i, VectorAssembler(inputCols = ["DayofMonth", "DayOfWeek", "OriginAirportID", "DestAirportID", "DepDelay"], outputCol="features"))
  lr.insert(i, LogisticRegression(labelCol="label", featuresCol="features"))
  pipeline.insert(i, Pipeline(stages=[assembler[i], lr[i]]))

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
# MAGIC 
# MAGIC #### Reference:
# MAGIC 1. https://docs.databricks.com/spark/latest/mllib/binary-classification-mllib-pipelines.html
# MAGIC 2. https://spark.apache.org/docs/2.1.0/ml-classification-regression.html#logistic-regression

# COMMAND ----------

# MAGIC %md ### Train Validation Split with Threshold parameters
# MAGIC Build the best model using TrainValidationSplit

# COMMAND ----------

# MAGIC %md (1) The first combination of parameters with (regParam: [0.01, 0.5]), (threshold: [0.30, 0.35]), (maxIter: [1, 5])

# COMMAND ----------

# define list of models made from Train Validation Split and Cross Validation
model = []

# COMMAND ----------

# params refered to the reference above
paramGrid = (ParamGridBuilder() \
             .addGrid(lr[0].regParam, [0.01, 0.5, 2.0]) \
             .addGrid(lr[0].threshold, [0.30, 0.35]) \
             .addGrid(lr[0].maxIter, [1, 5]) \
             .build())

# COMMAND ----------

tvs = TrainValidationSplit(estimator=pipeline[0], evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
# the first best model
model.insert(0, tvs.fit(train))

# COMMAND ----------

# MAGIC %md ### Train Validation Split with elastic-net parameters
# MAGIC 
# MAGIC ### Question 2 (5%): 
# MAGIC (2) Complete the second combination of parameters with (regParam: [0.01, 0.5, 2.0]), (elasticNetParam: [0.0, 0.5, 1]), (maxIter: [1, 5])

# COMMAND ----------

# TODO: params refered to the reference above
paramGrid2 = (ParamGridBuilder() \
             .addGrid(lr[1].regParam, [0.01, 0.5, 2.0]) \
             .addGrid(lr[1].elasticNetParam, [0.0, 0.5, 1]) \
             .addGrid(lr[1].maxIter, [1, 5]) \
             .build())


# COMMAND ----------

tvs2 = TrainValidationSplit(estimator=pipeline[1], evaluator=BinaryClassificationEvaluator(), estimatorParamMaps=paramGrid2, trainRatio=0.8)

# the second best model
model.insert(1, tvs2.fit(train))

# COMMAND ----------

# MAGIC %md ### Cross Validator with elastic net parameters
# MAGIC Build the best model using Cross Validator
# MAGIC 
# MAGIC ### Question 3 (5%)
# MAGIC (3) The combination of parameters with (regParam: [0.01, 0.5, 2.0]), (elasticNetParam: [0.0, 0.5, 1]), (maxIter: [1, 5])

# COMMAND ----------

# TODO: params refered to the reference above
paramGridCV = (ParamGridBuilder() \
             .addGrid(lr[2].regParam, [0.01, 0.5, 2.0]) \
             .addGrid(lr[2].elasticNetParam, [0.0, 0.5, 1]) \
             .addGrid(lr[2].maxIter, [1, 5]) \
             .build())

# COMMAND ----------

# TODO: K = 2 you may test it with 5, 10
# K=2, 3, 5, 
# K= 10 takes too long
cv = CrossValidator(estimator=pipeline[2], evaluator=BinaryClassificationEvaluator(), \
                    estimatorParamMaps=paramGridCV, numFolds=5)

# the third best model
model.insert(2, cv.fit(train))

# COMMAND ----------

# MAGIC %md ### Test the Model
# MAGIC Now you're ready to apply the model to the test data.

# COMMAND ----------

# list prediction
prediction = [] 
predicted = []
for i in range(3):
  prediction.insert(i, model[i].transform(test))
  predicted.insert(i, prediction[i].select("features", "prediction", "probability", "trueLabel"))
  predicted[i].show(30)

# COMMAND ----------

# MAGIC %md ### Compute Confusion Matrix Metrics
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

# MAGIC %md ### Question 4 (5%): Complete Fill-In to calculate Recall

# COMMAND ----------

# TODO: Complete the following [Fill-In] to calculate Recall
for i in range(3):
  tp = float(predicted[i].filter("prediction == 1.0 AND truelabel == 1").count())
  fp = float(predicted[i].filter("prediction == 1.0 AND truelabel == 0").count())
  tn = float(predicted[i].filter("prediction == 0.0 AND truelabel == 0").count())
  fn = float(predicted[i].filter("prediction == 0.0 AND truelabel == 1").count())
  metrics = spark.createDataFrame([
      ("TP", tp),
      ("FP", fp),
      ("TN", tn),
      ("FN", fn),
      ("Precision", tp / (tp + fp)),
      ("Recall", tp / (tp + fn))],["metric", "value"])
  metrics.show()

# COMMAND ----------

# MAGIC %md ### Review the Area Under ROC
# MAGIC Another way to assess the performance of a classification model is to measure the area under a ROC curve for the model. the spark.ml library includes a **BinaryClassificationEvaluator** class that you can use to compute this.

# COMMAND ----------

evaluator = []
for i in range(3):
  evaluator.insert(i, BinaryClassificationEvaluator(labelCol="trueLabel", rawPredictionCol="prediction", metricName="areaUnderROC"))
  aur = evaluator[i].evaluate(prediction[i])
  print ("AUR ", i, " = ", aur)

# COMMAND ----------

# MAGIC %md ### Question 5 (5%): Compare the three best models from 2 TrainSplitValidation and 1 CrossValidation. 
# MAGIC 
# MAGIC Explain which model you would choose and why in terms of the speed and accuracy (TP, FN, FP). Answer should be presented in [Fill-In] and it should be the maximum 3 lines - if the answer becomes more than 4 lines, you will get 0 point. You need to double click this cell to type in your answer and click once on another cell to complete the answer.
# MAGIC 
# MAGIC ANS) I would choose the second model because it is fast due to using Train Validation Split. Morever, it is as accurate as the third model which used Cross validation split. The nuber of True Postives(TP), False Negatives(FN), False Postives(FP) are the same. This means that the precsion, recall, and the AUC are also the same. The fact that the model is faster and as accurate as the third model makes it the best model for the project.

# COMMAND ----------

# MAGIC %md ### Question 6 (15%): Improve the AUR 
# MAGIC 1. assuming you don't change the data set and the rate of traing/testing. 
# MAGIC 1. You should get the highest AUR 0.837 from your solution. If you improve your PySpark code, you will get 1.5 point for each 0.01 increase of AUR.  
# MAGIC 1. Explain How you improve it - You need to add markdown cell above each of your PySpark cell to explain why you add the cell. Each Mark down cell should not be more than 3 lines.

# COMMAND ----------

# MAGIC %md ### Question 7 (10%): Export this to python code 
# MAGIC 1. Then, run it with spark-submit
# MAGIC 1. Take a screenshot of the result, that is, your AUR

# COMMAND ----------

# MAGIC %md You may build an experiment in Azure ML Studio to get hints to add the function of data engineering or google to find out some hints how others have done. The following is the reference that you may look at as well
# MAGIC 
# MAGIC ### References to improve the accuracy
# MAGIC 1. Extracting, transforming and selecting features, https://spark.apache.org/docs/latest/ml-features.html
# MAGIC 1. Basic data preparation in Pyspark, Normalizing and Scaling,  http://bit.ly/2Ihs6Wa
# MAGIC 1. Machine Learning with PySpark and ML a Binary Classification Problem, http://bit.ly/2Zb20tg

# COMMAND ----------

# MAGIC %md ANS) I used the GBTClassifier model for my prediction, because this model can generalize the data in a more effective manner. I used MulticlassClassificationEvaluator for the evaluation of AUR because it allows me to categorize the data with more than two classes. 
# MAGIC The new AUR is : 0.9234249120979983

# COMMAND ----------

gbt = GBTClassifier(labelCol="label", featuresCol="features", maxIter=10)

catVect = VectorAssembler(inputCols = ["DayofMonth",  "OriginAirportID"], outputCol="catFeatures")
catIdx = VectorIndexer(inputCol = catVect.getOutputCol(), outputCol = "idxCatFeatures")

numVect = VectorAssembler(inputCols = ["DepDelay"], outputCol="numFeatures")
# number vector is normalized. This will make sure that extreme values will not hinder my model
minMax = MinMaxScaler(inputCol = numVect.getOutputCol(), outputCol="normFeatures")

gbtassembler = VectorAssembler(inputCols=["idxCatFeatures", "normFeatures"], outputCol="features")

gbtp = Pipeline(stages=[catVect, catIdx, numVect, minMax, gbtassembler, gbt])

paramGrid = (ParamGridBuilder()
             .addGrid(gbt.maxDepth,[2,3,4])
             .addGrid(gbt.minInfoGain,[0.0, 0.1, 0.2, 0.3])
             .addGrid(gbt.stepSize,[0.05, 0.1, 0.2, 0.4])
             .build())

gbt_tvs = TrainValidationSplit(estimator=gbtp, evaluator=MulticlassClassificationEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)

gbtModel = gbt_tvs.fit(train)
predictions = gbtModel.transform(test)
prediction.insert(3, predictions)
predicted.insert(3, prediction[3].select("features", "prediction", "trueLabel"))

gbt_evaluator =  MulticlassClassificationEvaluator(labelCol="trueLabel", predictionCol="prediction")
gbt_auc = gbt_evaluator.evaluate(prediction[3])

print("The new AUC is: ", gbt_auc)
