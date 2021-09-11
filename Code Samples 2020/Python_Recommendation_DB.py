# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark Collaborative Filtering in Databricks
# MAGIC 
# MAGIC ### Jongwook Woo (jwoo5@calstatela.edu), revised on 04/17/2020, 04/21/2019, 04/27/2018
# MAGIC Tested in Runtime 5.2 (Spark 2.4.5/2.4.0 Scala 2.11) of Databricks CE

# COMMAND ----------

# MAGIC %md ## Collaborative Filtering
# MAGIC Collaborative filtering is a machine learning technique that predicts ratings awarded to items by users.
# MAGIC 
# MAGIC ### Import the ALS class
# MAGIC In this exercise, you will use the Alternating Least Squares collaborative filtering algorithm to creater a recommender.

# COMMAND ----------

from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit, CrossValidator

from pyspark.sql import functions as F

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

# COMMAND ----------

# MAGIC %md ### Load Source Data
# MAGIC The source data for the recommender is in two files - one containing numeric IDs for movies and users, along with user ratings; and the other containing details of the movies.

# COMMAND ----------

# MAGIC %md Read csv file from DBFS (Databricks File Systems)
# MAGIC 
# MAGIC ### TODO 1: follow the direction to read your table after upload it to Data at the left frame
# MAGIC NOTE: See above for the data type - Set String for _CustomerName_ and rest of the fields should have _Int_  data type - and reference [1]
# MAGIC 1. After _ratings.csv_ file is added to the data of the left frame, create a table using the UI, especially, "Upload File"
# MAGIC 1. Click "Preview Table to view the table" and Select the option as _ratings.csv_ has a header as the first row: "First line is header"
# MAGIC 1. __Change the data type__ of the table columns: (userId: int, movieId: int, rating: double, timestamp: string)
# MAGIC 1. When you click on create table button, remember the table name, for example, _ratings_csv_

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/ratings.csv

# COMMAND ----------

# MAGIC %md ### TODO 5: Set the following to True when you run its Python code in Spark Submit CLI

# COMMAND ----------

IS_SPARK_SUBMIT_CLI = True
if IS_SPARK_SUBMIT_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# MAGIC %md ### TODO 2: Assign the table name to data, which is created at TODO 1, using Spark SQL 
# MAGIC #### _spark.sql("SELECT * FROM ratings_csv")_, 
# MAGIC ratings = spark.sql("SELECT * FROM ratings_csv")

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
    ratings = spark.read.csv('ratings.csv', inferSchema=True, header=True)
else:
    ratings = spark.sql("SELECT * FROM ratings_csv")

# COMMAND ----------

# MAGIC %md Read movies.csv file from DBFS (Databricks File Systems)
# MAGIC 
# MAGIC ### TODO 3: follow the direction to read your table after upload it to Data at the left frame
# MAGIC NOTE: See reference [1]
# MAGIC 1. After _movies.csv_ file is added to the data of the left frame, create a table using the UI, especially, "Upload File"
# MAGIC 1. Click "Preview Table to view the table" and Select the option as movies.csv_ has a header as the first row: "First line is header"
# MAGIC 1. __Change the data type__ of the table columns: (movieId: int, title: string, genres: string)
# MAGIC 1. When you click on create table button, remember the table name, for example, _movies_csv_

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/ratings.csv

# COMMAND ----------

# MAGIC %md ### TODO 4: Assign the table name to data, which is created at TODO 1, using Spark SQL 
# MAGIC #### _spark.sql("SELECT * FROM movies_csv")_, 
# MAGIC movies = spark.sql("SELECT * FROM movies_csv")

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
    movies = spark.read.csv('movies.csv', inferSchema=True, header=True)
else:
    movies = spark.sql("SELECT * FROM movies_csv")

# Load the source data in Microsoft Azure
#ratings = spark.read.csv('wasb:///data/ratings.csv', inferSchema=True, header=True)


# COMMAND ----------

movies.show(5)

# COMMAND ----------

ratings.join(movies, "movieId").show()

# COMMAND ----------

# MAGIC %md ### ml-100k: https://grouplens.org/datasets/movielens/100k/
# MAGIC 943 users and 1682 items for 100,000 ratings
# MAGIC 
# MAGIC ### ml-1M: https://grouplens.org/datasets/movielens/1m/
# MAGIC - UserIDs range between 1 and 6040 
# MAGIC - MovieIDs range between 1 and 3952

# COMMAND ----------

# movies_unique = spark.sql("SELECT count(unique moviesId) FROM movies")
movies.select("movieId").distinct().count()
# movies.groupBy("movieId").count().orderBy().show()


# COMMAND ----------

ratings.select("userId").distinct().count()


# COMMAND ----------

# MAGIC %md ### Prepare the Data
# MAGIC To prepare the data, split it into a training set and a test set.

# COMMAND ----------

data = ratings.select("userId", "movieId", "rating")
splits = data.randomSplit([0.7, 0.3])
train = splits[0].withColumnRenamed("rating", "label")
test = splits[1].withColumnRenamed("rating", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print ("Training Rows:", train_rows, " Testing Rows:", test_rows)

# COMMAND ----------

# MAGIC %md ### Build the Recommender
# MAGIC In ALS, users and products (movies) are described by a small set of latent features (factors) that can be used to predict missing entries.
# MAGIC 
# MAGIC #### Latent Features
# MAGIC latent features might be things like properties and genre of the movies: action, romance, comedy...
# MAGIC We can use the features to produce some sort of algorithm (**ALS**) to intelligently calculate ratings 
# MAGIC 
# MAGIC The ALS class is an estimator, so you can use its **fit** method to traing a model, or you can include it in a pipeline. Rather than specifying a feature vector and as label, the ALS algorithm requries a numeric user ID, item ID, and rating.

# COMMAND ----------

als = ALS(userCol="userId", itemCol="movieId", ratingCol="label")
#als = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="label")
#model = als.fit(train)

# COMMAND ----------

# MAGIC %md #### Add paramGrid and Validation: JWoo5

# COMMAND ----------

paramGrid = ParamGridBuilder() \
                    .addGrid(als.rank, [1, 5]) \
                    .addGrid(als.maxIter, [5, 10]) \
                    .addGrid(als.regParam, [0.3, 0.1]) \
                    .addGrid(als.alpha, [2.0,3.0]) \
                    .build()



# COMMAND ----------

# MAGIC %md ### It takes about _1 hour_ at Databricks CE with _3 regParams_
# MAGIC #### _15 minutes_ with _2 regParams_
# MAGIC #### Using Oracle BDCE (3 Nodes, 48 OCPUs, 720 GB Memory, 1.1 TB Storage), it takes _5 minutes_ to build a model
# MAGIC In the cell abive, we will use only _2 regParams_
# MAGIC ```
# MAGIC paramGrid = ParamGridBuilder() \
# MAGIC                     .addGrid(als.rank, [1, 5]) \
# MAGIC                     .addGrid(als.maxIter, [5, 10]) \
# MAGIC                     .addGrid(als.regParam, [0.3, 0.1, 0.01]) \
# MAGIC                     .addGrid(als.alpha, [2.0,3.0]) \
# MAGIC                     .build()
# MAGIC ```

# COMMAND ----------

# MAGIC %md ### To build a general model, _TrainValidationSplit_ is adopted as it is much faster than _CrossValidator_
# MAGIC You can run a code with __CrossValidator__ instead as follows:
# MAGIC ```
# MAGIC cv = CrossValidator(estimator=alsImplicit, estimatorParamMaps=paramGrid, evaluator=RegressionEvaluator())
# MAGIC ```

# COMMAND ----------

cv = TrainValidationSplit(estimator=als, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
#cv = CrossValidator(estimator=alsImplicit, estimatorParamMaps=paramGrid, evaluator=RegressionEvaluator())
model = cv.fit(train)

# COMMAND ----------

# MAGIC %md ### Test the Recommender
# MAGIC Now that you've trained the recommender, you can see how accurately it predicts known ratings in the test set.

# COMMAND ----------

prediction = model.transform(test)

# Remove NaN values from prediction (due to SPARK-14489) [1]
prediction = prediction.filter(prediction.prediction != float('nan'))

# Round floats to whole numbers
prediction = prediction.withColumn("prediction", F.abs(F.round(prediction["prediction"],0)))

prediction.join(movies, "movieId").select("userId", "title", "prediction", "trueLabel").show(100, truncate=False)

# COMMAND ----------

# MAGIC %md #### RegressionEvaluator
# MAGIC Calculate RMSE using RegressionEvaluator.
# MAGIC 
# MAGIC __NOTE:__ make sure to set [predictionCol="prediction"]

# COMMAND ----------

# RegressionEvaluator: predictionCol="prediction", metricName="rmse"
evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(prediction)
print ("Root Mean Square Error (RMSE):", rmse)

# COMMAND ----------

# MAGIC %md ### Two types of user preferences:
# MAGIC 
# MAGIC __Explicit preference__ (also referred as "Explicit feedback"), such as "rating" given to item by users. Default for ALS
# MAGIC 
# MAGIC __Implicit preference__ (also referred as "Implicit feedback"), such as "view" and "buy" history.

# COMMAND ----------

# MAGIC %md ### ALS model in implicit type
# MAGIC If the rating matrix is derived from another source of information (i.e. it is inferred from other signals), you can set implicitPrefs to True to get better results. It is **not** the case of this data set.
# MAGIC 
# MAGIC Build and Train ALS model with "implicitPrefs=True"

# COMMAND ----------

als_implicit = ALS(userCol="userId", itemCol="movieId", ratingCol="label", implicitPrefs=True)
#als_implicit = ALS(maxIter=5, regParam=0.01, userCol="userId", itemCol="movieId", ratingCol="label", implicitPrefs=True)
#model_implicit = als_implicit.fit(train)

# COMMAND ----------

paramGrid = ParamGridBuilder() \
                    .addGrid(als_implicit.rank, [1, 5]) \
                    .addGrid(als_implicit.maxIter, [5, 10]) \
                    .addGrid(als_implicit.regParam, [0.3, 0.1, 0.01]) \
                    .addGrid(als_implicit.alpha, [2.0,3.0]) \
                    .build()


# COMMAND ----------

cv = TrainValidationSplit(estimator=als_implicit, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, trainRatio=0.8)
#cv = CrossValidator(estimator=als_implicit, estimatorParamMaps=paramGrid, evaluator=RegressionEvaluator())
model_implicit = cv.fit(train)

# COMMAND ----------

prediction_implicit = model_implicit.transform(test)

# Remove NaN values from prediction (due to SPARK-14489) [1]
prediction_implicit = prediction_implicit.filter(prediction_implicit.prediction != float('nan'))

# Round floats to whole numbers
prediction_implicit = prediction_implicit.withColumn("prediction", F.abs(F.round(prediction_implicit["prediction"],0)))


prediction_implicit.join(movies, "movieId").select("userId", "title", "prediction", "trueLabel").show(100, truncate=False)

# COMMAND ----------

# RegressionEvaluator: predictionCol="prediction", metricName="rmse"
evaluator_implicit = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
rmse_implicit = evaluator_implicit.evaluate(prediction_implicit)
print ("ImplicitRoot Mean Square Error (RMSE):", rmse_implicit)

# COMMAND ----------

# MAGIC %md ## TODO 5: Execute the ipynb code as Python code using Spark Submit command
# MAGIC 1. Export this ipynb code to Python code - refer to the previous lab tutorial
# MAGIC 1. __scp__ the python code to your Hadoop/Spark server, 
# MAGIC ```
# MAGIC $ scp xxx.py jwoo5@129.xxx.xx.160:~/
# MAGIC ```
# MAGIC 1. Upload data files to your Hadoop/Spark server: customers.csv, movies.csv, ratings.csv
# MAGIC 1. Execute the python code at your Hadoop/Spark server using spark-submit command
# MAGIC ```
# MAGIC spark-submit xxx.py
# MAGIC ```

# COMMAND ----------

# MAGIC %md The data used in this exercise describes 5-star rating activity from [MovieLens](http://movielens.org), a movie recommendation service. It was created by GroupLens, a research group in the Department of Computer Science and Engineering at the University of Minnesota, and is used here with permission.
# MAGIC 
# MAGIC This dataset and other GroupLens data sets are publicly available for download at <http://grouplens.org/datasets/>.
# MAGIC 
# MAGIC For more information, see F. Maxwell Harper and Joseph A. Konstan. 2015. [The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4, Article 19 (December 2015)](http://dx.doi.org/10.1145/2827872)
# MAGIC 
# MAGIC **Reference**
# MAGIC 1. Training with Implicit Preference (Recommendation), https://predictionio.apache.org/templates/recommendation/training-with-implicit-preference/
# MAGIC 1. Predicting Song Listens Using Apache Spark, https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/3175648861028866/48824497172554/657465297935335/latest.html

# COMMAND ----------


