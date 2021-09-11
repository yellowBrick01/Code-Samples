# Databricks notebook source
# MAGIC %md ## CIS5560: PySpark LDA Text  Analysis in Databricks
# MAGIC 
# MAGIC ### Jongwook Woo (jwoo5@calstatela.edu), 04/19/2020, 04/21/2019, 05/02/2018
# MAGIC Tested in Runtime 5.2 (Spark 2.4.5/2.4.0 Scala 2.11) of Databricks CE

# COMMAND ----------

# MAGIC %md ## Text Analysis using Latent Dirichlet Allocation (LDA)
# MAGIC In this lab, you will create a classification model that performs sentiment analysis of tweets.
# MAGIC ### Import Spark SQL and Spark ML Libraries
# MAGIC 
# MAGIC First, import the libraries you will need:

# COMMAND ----------

from pyspark.ml.feature import HashingTF, IDF, Tokenizer, CountVectorizer
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA, BisectingKMeans
from pyspark.sql.functions import monotonically_increasing_id

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

import re

# COMMAND ----------

# MAGIC %md ### Load Source Data
# MAGIC Now load the tweets data into a DataFrame. This data consists of tweets that have been previously captured and classified as positive or negative.

# COMMAND ----------

# True when to create Python soure code to run with spark-submit 
IS_SPARK_SUBMIT_CLI = True

if IS_SPARK_SUBMIT_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

# MAGIC %md ### Code for Databricks to read airlines.csv file 
# MAGIC #### Create a table: _airlines_csv_ with _airlines.csv_ file
# MAGIC all colums are of String except (id: int, rating: int, value: int)
# MAGIC #### TODO 1: Need to upload airlines.csv file to DBFS (Databricks File Systems) 
# MAGIC #### TODO 2: Need to upload airlines.csv file to HDFS for Python using spark-submit CLI

# COMMAND ----------

# Databricks needs to create a table: airlines_csv with airlines.csv file
# all colums are of String except (id: int, rating: int, value: int)
if IS_SPARK_SUBMIT_CLI:
    airlines = spark.read.csv('airlines.csv', inferSchema=True, header=True)
else:
    airlines = sqlContext.sql("select * from airlines_csv")
    
airlines.show(5)


# COMMAND ----------

# MAGIC %md ### Assign the variable df_data_x to rawdata, which is created at TODO 1 
# MAGIC #### For example, when it is _df_data_2_, 
# MAGIC rawdata = airlines

# COMMAND ----------

# Microsoft Azure 
# tweets_csv = spark.read.csv('wasb:///data/tweets.csv', inferSchema=True, header=True)
# tweets_csv.show(truncate = False)
rawdata = airlines

# Show rawdata (as DataFrame)
rawdata.show(5)

# COMMAND ----------

# MAGIC %md Create columns _uid_ automatically and _year_month_ columns from _date_

# COMMAND ----------

rawdata = rawdata.fillna({'review': ''})                               # Replace nulls with blank string

# Add Unique ID
rawdata = rawdata.withColumn("uid", monotonically_increasing_id())     # Create Unique ID

# Generate YYYY-MM variable
rawdata = rawdata.withColumn("year_month", rawdata.date.substr(1,7))

# Show rawdata (as DataFrame)
rawdata.show(10)

# COMMAND ----------

# MAGIC %md Print data types

# COMMAND ----------

# Print data types
for type in rawdata.dtypes:
    print (type)


# COMMAND ----------

# MAGIC %md Change data type of the column _rating_ to integer

# COMMAND ----------

target = rawdata.select(rawdata['rating'].cast(IntegerType()))
target.dtypes

# COMMAND ----------

# MAGIC %md ## Text Pre-processing (consider using one or all of the following):
# MAGIC ###   
# MAGIC 1. Remove common words (with stoplist)
# MAGIC 1. Handle punctuation
# MAGIC 1. lowcase/upcase
# MAGIC 1. Stemming
# MAGIC 1. Part-of-Speech Tagging (nouns, verbs, adj, etc.)

# COMMAND ----------

def cleanup_text(record):
    text  = record[8]
    uid   = record[9]
    words = text.split()
    
    # Default list of Stopwords
    stopwords_core = ['a', u'about', u'above', u'after', u'again', u'against', u'all', u'am', u'an', u'and', u'any', u'are', u'arent', u'as', u'at', 
    u'be', u'because', u'been', u'before', u'being', u'below', u'between', u'both', u'but', u'by', 
    u'can', 'cant', 'come', u'could', 'couldnt', 
    u'd', u'did', u'didn', u'do', u'does', u'doesnt', u'doing', u'dont', u'down', u'during', 
    u'each', 
    u'few', 'finally', u'for', u'from', u'further', 
    u'had', u'hadnt', u'has', u'hasnt', u'have', u'havent', u'having', u'he', u'her', u'here', u'hers', u'herself', u'him', u'himself', u'his', u'how', 
    u'i', u'if', u'in', u'into', u'is', u'isnt', u'it', u'its', u'itself', 
    u'just', 
    u'll', 
    u'm', u'me', u'might', u'more', u'most', u'must', u'my', u'myself', 
    u'no', u'nor', u'not', u'now', 
    u'o', u'of', u'off', u'on', u'once', u'only', u'or', u'other', u'our', u'ours', u'ourselves', u'out', u'over', u'own', 
    u'r', u're', 
    u's', 'said', u'same', u'she', u'should', u'shouldnt', u'so', u'some', u'such', 
    u't', u'than', u'that', 'thats', u'the', u'their', u'theirs', u'them', u'themselves', u'then', u'there', u'these', u'they', u'this', u'those', u'through', u'to', u'too', 
    u'under', u'until', u'up', 
    u'very', 
    u'was', u'wasnt', u'we', u'were', u'werent', u'what', u'when', u'where', u'which', u'while', u'who', u'whom', u'why', u'will', u'with', u'wont', u'would', 
    u'y', u'you', u'your', u'yours', u'yourself', u'yourselves']
    
    # Custom List of Stopwords - Add your own here
    stopwords_custom = ['']
    stopwords = stopwords_core + stopwords_custom
    stopwords = [word.lower() for word in stopwords]    
    
    text_out = [re.sub('[^a-zA-Z0-9]','',word) for word in words]                                       # Remove special characters
    text_out = [word.lower() for word in text_out if len(word)>2 and word.lower() not in stopwords]     # Remove stopwords and words under X length
    return text_out



# COMMAND ----------

# MAGIC %md ## Cleaning Text

# COMMAND ----------

udf_cleantext = udf(cleanup_text , ArrayType(StringType()))
clean_text = rawdata.withColumn("words", udf_cleantext(struct([rawdata[x] for x in rawdata.columns])))

#tokenizer = Tokenizer(inputCol="description", outputCol="words")
#wordsData = tokenizer.transform(text)

# COMMAND ----------

#clean_text.show(2)

# COMMAND ----------

# MAGIC %md ## Generate TFIDF and Vectorize it

# COMMAND ----------

# Term Frequency Vectorization  - Option 1 (Using hashingTF): 
'''hashingTF = HashingTF(inputCol="words", outputCol="rawFeatures", numFeatures=20)
featurizedData = hashingTF.transform(clean_text)
'''
# Term Frequency Vectorization  - Option 2 (CountVectorizer)    : 
cv = CountVectorizer(inputCol="words", outputCol="rawFeatures", vocabSize = 1000)
cvmodel = cv.fit(clean_text)
featurizedData = cvmodel.transform(clean_text)

vocab = cvmodel.vocabulary
vocab_broadcast = sc.broadcast(vocab)


# COMMAND ----------

# MAGIC %md ### Term Frequency (TF)
# MAGIC ### Inverse Document Frequency (IDF)

# COMMAND ----------

idf = IDF(inputCol="rawFeatures", outputCol="features")
idfModel = idf.fit(featurizedData)
rescaledData = idfModel.transform(featurizedData)

# COMMAND ----------

# MAGIC %md ## LDA Clustering 
# MAGIC ### Generate 25 Data-Driven Topics:

# COMMAND ----------

# Generate 25 Data-Driven Topics:
lda = LDA(k=25, seed=123, optimizer="em", featuresCol="features")

ldamodel = lda.fit(rescaledData)

#model.isDistributed()
#model.vocabSize()

ldatopics = ldamodel.describeTopics()
#ldatopics.show(25)

# COMMAND ----------

# MAGIC %md ### LDA Clustering - Find Data-driven Topics

# COMMAND ----------



def map_termID_to_Word(termIndices):
    words = []
    for termID in termIndices:
        words.append(vocab_broadcast.value[termID])
    
    return words

udf_map_termID_to_Word = udf(map_termID_to_Word , ArrayType(StringType()))
ldatopics_mapped = ldatopics.withColumn("topic_desc", udf_map_termID_to_Word(ldatopics.termIndices))
ldatopics_mapped.select(ldatopics_mapped.topic, ldatopics_mapped.topic_desc).show(50,False)


# COMMAND ----------

# MAGIC %md ## Topics you can guess from the above result:
# MAGIC 1. Topic 0: Seating Concerns (more legroom, exit aisle, extra room)
# MAGIC 1. Topic 1: Vegas Trips (including upgrades)
# MAGIC 1. Topic 7: Carryon Luggage, Overhead Space Concerns
# MAGIC 1. Topic 12: Denver Delays, Mechanical Issues

# COMMAND ----------

# MAGIC %md ### Combine Data-Driven Topics with the Original Airlines Dataset

# COMMAND ----------

ldaResults = ldamodel.transform(rescaledData)

ldaResults.select('id','airline','date','cabin','rating','words','features','topicDistribution').show()

# COMMAND ----------

# MAGIC %md ### Breakout LDA Topics for Modeling and Reporting

# COMMAND ----------

def breakout_array(index_number, record):
    vectorlist = record.tolist()
    return vectorlist[index_number]

udf_breakout_array = udf(breakout_array, FloatType())

# Extract document weights for Topics 12 and 20
enrichedData = ldaResults                                                                   \
        .withColumn("Topic_12", udf_breakout_array(lit(12), ldaResults.topicDistribution))  \
        .withColumn("topic_20", udf_breakout_array(lit(20), ldaResults.topicDistribution))            

enrichedData.select('id','airline','date','cabin','rating','words','features','topicDistribution','Topic_12','Topic_20').show()

enrichedData.agg(max("Topic_12")).show()

# COMMAND ----------

enrichedData.createOrReplaceTempView("enrichedData")

# COMMAND ----------

# MAGIC %md ## Visualize Airline Volume and Average Rating Trends (by Date)

# COMMAND ----------

#topics = enrichedData.select('id','airline','date','year_month', 'rating', 'topic_12', 'topic_20').sort(desc("date"))
#topics = topics.filter(col('airline') == 'Delta Air Lines')
#topics = topics.filter(topics.airline.like("%Delta Air Lines%"))

# COMMAND ----------

topics = enrichedData.select('id','airline','date','rating').where(col("airline").isin(["Delta Air Lines", "US Airways", "Southwest Airlines", "American Airlines", "United Airlines"])).sort(desc("date"))

# topics = enrichedData.select('id','airline','date','year_month', 'rating', 'topic_12', 'topic_20').where((col("date") >= "1-Jan-15") & (col("airline").isin(["Delta Air Lines", "US Airways", "Southwest Airlines", "American Airlines", "United Airlines"]))).sort(desc("date"))

topics.show(5)


# COMMAND ----------

topics.createOrReplaceTempView("topicsView")

# COMMAND ----------

# MAGIC %md ### The following _SQL_ and _Display_ only for DatabricksI
# MAGIC ```
# MAGIC if ~IS_SPARK_SUBMIT_CLI:
# MAGIC     display(topics)
# MAGIC ```

# COMMAND ----------

# MAGIC %md ## Show Bar Chart of 5 airlines 
# MAGIC 
# MAGIC ### For Databricks: 
# MAGIC 1. Select chart 
# MAGIC 1. Select __Plot Options__: _Bar Chart_ with (Key: Date, Series Grouping: Airline, Values: Rating, Aggregation: AVG, Grouped)

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from topicsView

# COMMAND ----------

# Only for Databricks
'''
%sql
SELECT id, airline, date, year_month, rating, topic_12, topic_20 FROM enrichedData where airline = "${item=Delta Air Lines,Delta Air Lines|US Airways|Southwest Airlines|American Airlines|United Airlines}" order by date
'''

# COMMAND ----------

# MAGIC %md ## Select Topic 12 and 20
# MAGIC 1. Topic 12:  denver, mechanical, hours, delayed, airport, connecting,...
# MAGIC 1. Topic 20: personal, configuration, honolulu, entertainment, phx,...

# COMMAND ----------

two_topics = enrichedData.select('id','airline','date','year_month', 'rating', 'topic_12', 'topic_20').where(col("airline").isin(["Delta Air Lines", "US Airways", "Southwest Airlines", "American Airlines", "United Airlines"])).sort(desc("date"))

# COMMAND ----------

two_topics.createOrReplaceTempView("two_topics_view")

# COMMAND ----------

# MAGIC %md ### Only for Databricks Not for PySpark CLI
# MAGIC ```
# MAGIC if ~IS_SPARK_SUBMIT_CLI:
# MAGIC     display(two_topics)
# MAGIC ```

# COMMAND ----------

# MAGIC %md ### TODO 3.a: Options for Plot: 
# MAGIC #### Bar Chart
# MAGIC Keys (Airline), Series Grouping (Airline), Values (rating), Aggregation (AVG), Stacked

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from two_topics_view

# COMMAND ----------

# MAGIC %md ### TODO 3.a: Write the dataframe to HDFS as of csv (or parquet) in Spark CLI
# MAGIC 1. Check out if the file is created in HDFS: 
# MAGIC ```
# MAGIC hdfs dfs -ls lda
# MAGIC ```
# MAGIC 1. Merge the multiple files at HDFS to csv file of your local file systems
# MAGIC ```
# MAGIC hdfs dfs -cat ./lda/* > twoTopics.csv
# MAGIC ls -al
# MAGIC ...
# MAGIC -rw-rw-r--.  1 jwoo5 jwoo5    64060 Apr 20 00:40 twoTopics.csv
# MAGIC ```
# MAGIC 1. Check out if the file created has the value correctly using _cat_ and _head_
# MAGIC ```
# MAGIC cat twoTopics.csv | head -5
# MAGIC ...
# MAGIC 10505,Southwest Airlines,9-May-14,9-May-1,0,0.060210884,0.023569606
# MAGIC 10746,American Airlines,9-May-14,9-May-1,10,0.035496533,0.03617496
# MAGIC ```
# MAGIC 1. Check out if the file created has the value correctly using _cat_ and _tail_:
# MAGIC ```
# MAGIC cat twoTopics.csv | tail -n 5
# MAGIC ...
# MAGIC 10922,United Airlines,1-Apr-14,1-Apr-1,1,0.025105191,0.090657696
# MAGIC 10923,United Airlines,1-Apr-14,1-Apr-1,1,0.017568659,0.053080194
# MAGIC ```
# MAGIC 1. Download the csv file using _scp_ to your desktop computer and visualize it in excel to create a chart similar to __TODO 3.a__
# MAGIC 
# MAGIC #### NOTE: files in HDFS cannot be re-written again with the same file name as Hadoop is Write-Once systems. 
# MAGIC 1. You need to change the file name
# MAGIC 1. You need to delete the exsting file in HDFS

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
    #two_topics.write.parquet("./twoTopics.parquet")
    two_topics.write.csv('./lda')

# COMMAND ----------

# MAGIC %md ## Show all reviews related to Topic 12

# COMMAND ----------

# MAGIC %md ### You can run the following code only in Databricks
# MAGIC ```
# MAGIC %sql
# MAGIC SELECT ID, DATE, AIRLINE, REVIEW, TOPIC_12 FROM ENRICHEDDATA WHERE TOPIC_12 >= 0.25 ORDER BY TOPIC_12 DESC
# MAGIC ```

# COMMAND ----------

# For PySpark
review = enrichedData.select('id','airline','date','review', 'topic_12').where(col("TOPIC_12")>=0.25).sort(desc("TOPIC_12"))

# COMMAND ----------

review.select(col("TOPIC_12"), 'review').take(2)

# COMMAND ----------

review.createOrReplaceTempView("review_view")

# COMMAND ----------

# MAGIC %md ### Not for PySpark CLI
# MAGIC ```
# MAGIC if ~IS_SPARK_SUBMIT_CLI:
# MAGIC     display(review)
# MAGIC ```

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from review_view

# COMMAND ----------

# MAGIC %md ### TODO 4: Create all reviews related to _Topic 20_ simliar to Topic 12 above

# COMMAND ----------

# MAGIC %md ### TODO 5: Make this code run using _spark-submit_ in __PySpark CLI__.
# MAGIC Don't forget to change the value to _True_ of __IS_SPARK_SUBMIT_CLI__

# COMMAND ----------

# MAGIC %md ### Reference
# MAGIC 1. https://community.hortonworks.com/articles/84781/spark-text-analytics-uncovering-data-driven-topics.html
# MAGIC 1. https://ibm-watson-data-lab.github.io/pixiedust/install.html
# MAGIC 1. https://dataplatform.ibm.com/docs/content/pixiedust/writeviz.html
# MAGIC 1. https://spark.apache.org/docs/2.2.0/ml-clustering.html#latent-dirichlet-allocation-lda
