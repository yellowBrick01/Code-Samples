# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark Clustering in Databricks
# MAGIC 
# MAGIC ### Jongwook Woo (jwoo5@calstatela.edu), revised on 04/17/2020, 04/21/2019, 04/27/2018
# MAGIC Tested in Runtime 5.2 (Spark 2.4.5/2.4.0 Scala 2.11) of Databricks CE

# COMMAND ----------

# MAGIC %md ## Clustering
# MAGIC In this exercise, you will use K-Means clustering to segment customer data into five clusters.
# MAGIC 
# MAGIC ### Import the Libraries
# MAGIC You will use the **KMeans** class to create your model. This will require a vector of features, so you will also use the **VectorAssembler** class.

# COMMAND ----------

from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

# COMMAND ----------

# MAGIC %md ### Load Source Data
# MAGIC The source data for your clusters is in a comma-separated values (CSV) file, and incldues the following features
# MAGIC 
# MAGIC __NOTE:__ Set String for _CustomerName_ and rest of the fields should have _Int_  data type
# MAGIC - CustomerName (String): The custome's name
# MAGIC - Age (Int): The customer's age in years
# MAGIC - MaritalStatus (Int): The custtomer's marital status (1=Married, 0 = Unmarried)
# MAGIC - IncomeRange (Int): The top-level for the customer's income range (for example, a value of 25,000 means the customer earns up to 25,000)
# MAGIC - Gender (Int): A numeric value indicating gender (1 = female, 2 = male)
# MAGIC - TotalChildren (Int): The total number of children the customer has
# MAGIC - ChildrenAtHome (Int): The number of children the customer has living at home.
# MAGIC - Education (Int): A numeric value indicating the highest level of education the customer has attained (1=Started High School to 5=Post-Graduate Degree
# MAGIC - Occupation (Int): A numeric value indicating the type of occupation of the customer (0=Unskilled manual work to 5=Professional)
# MAGIC - HomeOwner (Int): A numeric code to indicate home-ownership (1 - home owner, 0 = not a home owner)
# MAGIC - Cars (Int): The number of cars owned by the customer.

# COMMAND ----------

# MAGIC %md Read csv file from DBFS (Databricks File Systems)
# MAGIC 
# MAGIC ### TODO 1: follow the direction to read your table after upload it to Data at the left frame
# MAGIC NOTE: See above for the data type - Set String for _CustomerName_ and rest of the fields should have _Int_  data type - and reference [1]
# MAGIC 1. After _customers.csv_ file is added to the data of the left frame, create a table using the UI, especially, "Upload File"
# MAGIC 1. Click "Preview Table to view the table" and Select the option as _customers.csv_ has a header as the first row: "First line is header"
# MAGIC 1. __Change the data type__ of the table columns as shown in the previous cell
# MAGIC 1. When you click on create table button, remember the table name, for example, _customers_csv_

# COMMAND ----------

# MAGIC %fs ls /FileStore/tables/customers.csv

# COMMAND ----------

# MAGIC %md ### TODO 4: Set the following to True when you run its Python code in Spark Submit CLI

# COMMAND ----------

IS_SPARK_SUBMIT_CLI = True
if IS_SPARK_SUBMIT_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# MAGIC %md ### TODO 2: Assign the table name to data, which is created at TODO 1, using Spark SQL 
# MAGIC #### _spark.sql("SELECT * FROM customers_csv")_, 
# MAGIC customers = spark.sql("SELECT * FROM customers_csv")

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
    customers = spark.read.csv('customers.csv', inferSchema=True, header=True)
else:
    customers= spark.sql("SELECT * FROM customers_csv")

# COMMAND ----------

customers.show(5)

# COMMAND ----------

customers.printSchema()

# COMMAND ----------

# MAGIC %md ### Create the K-Means Model
# MAGIC You will use the feaures in the customer data to create a Kn-Means model with a k value of 5. This will be used to generate 5 clusters.
# MAGIC __NOTE__: The predicted label should have the column name as __prediction__ in order to get used at __ClusteringEvaluator__ class.

# COMMAND ----------

assembler = VectorAssembler(inputCols = ["Age", "MaritalStatus", "IncomeRange", "Gender", "TotalChildren", "ChildrenAtHome", "Education", "Occupation", "HomeOwner", "Cars"], outputCol="features")
train = assembler.transform(customers)

knum = 5
# Make sure to set [predictionCol="prediction"]
kmeans = KMeans(featuresCol=assembler.getOutputCol(), predictionCol="prediction", k=knum, seed=0)
model = kmeans.fit(train)
print ("Model Created!")

# COMMAND ----------

# MAGIC %md ### Predict Clusters
# MAGIC Now that you have trained the model, you can use it to segemnt the customer data into 5 clusters and show each customer with their allocated cluster.

# COMMAND ----------

# data set does not need to be divided to train and test
predictions = model.transform(train)
predictions.groupBy("prediction").count().orderBy("prediction").show()

# COMMAND ----------

predictions.printSchema()

# COMMAND ----------

# Prediction is Cluster
predictions.select("features", "prediction").show(5)

# COMMAND ----------

predictions.select("CustomerName", "prediction").show(50)

# COMMAND ----------

# MAGIC %md ### Evaluation Clustering 
# MAGIC ClusteringEvaluator is easy to use but it is available since Spark 2.3.0. Databricks CE supports Spark 2.3.0 but DSX for Spark 2.1 as of April 27 2018
# MAGIC 
# MAGIC The silhouette value is a measure of how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The silhouette ranges from (minus 1) to +1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters. If most objects have a high value, then the clustering configuration is appropriate. If many points have a low or negative value, then the clustering configuration may have too many or too few clusters.
# MAGIC 
# MAGIC The silhouette can be calculated with any distance metric, such as the Euclidean distance or the Manhattan distance : https://en.wikipedia.org/wiki/Silhouette_(clustering)
# MAGIC 
# MAGIC ## TODO 3: make sure 
# MAGIC IS_SPARK230 = True

# COMMAND ----------

# > Spark 2.3.0?
IS_SPARK230 = False

if IS_SPARK230:
    from pyspark.ml.evaluation import ClusteringEvaluator

    # Evaluate clustering by computing Silhouette score
    #evaluator =  ClusteringEvaluator().setPredictionCol("cluster").setFeaturesCol("features").setMetricName("silhouette")
    evaluator = ClusteringEvaluator()
    #print evaluator.explainParams
    
    # Needs Parameters: prediction (of DoubleType values) and label (of float or double values)
    silhouette = evaluator.evaluate(predictions)
    # close to 1
    print("Silhouette with squared euclidean distance = " + str(silhouette))
else:
    # Previous Spark: Evaluate clustering by computing Within Set Sum of Squared Errors.
    wssse = model.computeCost(train)
    print("Within Set Sum of Squared Errors = " + str(wssse))



# COMMAND ----------

# MAGIC %md ### Get the Cluster Centers
# MAGIC The cluster centers are indicated as vector coordinates.

# COMMAND ----------

# Shows the result.
centers = model.clusterCenters()
print("Cluster Centers: ")
for center in centers:
    print(center)

# COMMAND ----------

# MAGIC %md ### Display the features of each cluster
# MAGIC Display by Cluster that is by the column: __prediction__

# COMMAND ----------

# Look at the features of each cluster

# define dictionary
customerCluster = {}
for i in range(0,knum):
    # Group by Cluster that is prediction
    tmp = predictions.select("Age", "MaritalStatus", "IncomeRange", "Gender", "TotalChildren", \
                                        "ChildrenAtHome", "Education", "Occupation", "HomeOwner", "Cars")\
                                    .where("prediction =" +  str(i))
    customerCluster[str(i)]= tmp
    print ("Cluster"+str(i))
    customerCluster[str(i)].show(10)

# COMMAND ----------

if ~IS_SPARK_SUBMIT_CLI:
  display(model)

# COMMAND ----------

# MAGIC %md ## TODO 4: Execute the ipynb code as Python code using Spark Submit command
# MAGIC 1. Go up to the above to change _IS_SPARK_SUBMIT_CLI_ to __True__.
# MAGIC 1. Export this ipynb code to Python code - refer to the previous lab tutorial
# MAGIC 1. __scp__ the python code to your Hadoop/Spark server, 
# MAGIC ```
# MAGIC $ scp xxx.py jwoo5@129.xxx.xx.160:~/
# MAGIC ```
# MAGIC 1. 1. Upload data files to your Hadoop/Spark server: customers.csv, movies.csv, ratings.csv
# MAGIC 1. Execute the python code at your Hadoop/Spark server using spark-submit command
# MAGIC ```
# MAGIC spark-submit xxx.py
# MAGIC ```

# COMMAND ----------

# MAGIC %md ### References
# MAGIC 1. https://spark.apache.org/docs/latest/ml-clustering.html
# MAGIC 1. https://spark.apache.org/docs/preview/ml-clustering.html
# MAGIC 1. Determining Optimal Clusters, https://uc-r.github.io/kmeans_clustering#silo
# MAGIC 1. Silhouette (clustering), https://en.wikipedia.org/wiki/Silhouette_(clustering)
# MAGIC 1. https://jaceklaskowski.gitbooks.io/mastering-apache-spark/content/spark-courses.html
