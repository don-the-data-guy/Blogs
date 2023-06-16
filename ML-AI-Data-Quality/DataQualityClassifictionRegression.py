# Databricks notebook source
# MAGIC %sql
# MAGIC /* Step 1 Create table */
# MAGIC CREATE TABLE job_data (
# MAGIC     age INT,
# MAGIC     income INT,
# MAGIC     job_type VARCHAR(50)
# MAGIC );

# COMMAND ----------

# MAGIC %sql
# MAGIC /* Step 2 Load data to table */
# MAGIC INSERT INTO job_data (age, income, job_type)
# MAGIC VALUES
# MAGIC     (25, 50000, 'Engineer'),
# MAGIC     (30, 55000, 'Doctor'),
# MAGIC     (40, 80000, 'Lawyer'),
# MAGIC     (22, 40000, NULL),
# MAGIC     (35, 65000, 'Engineer'),
# MAGIC     (28, 48000, NULL),
# MAGIC     (45, 90000, 'Lawyer'),
# MAGIC     (50, 92000, 'Doctor'),
# MAGIC     (33, 60000, NULL);

# COMMAND ----------

# Step 3: Load the data
query = "SELECT age, income, job_type from job_data"

# df here should be replaced by your actual DataFrame
df = spark.sql(query)

# COMMAND ----------

# Step 4: Process the data
df_known = df.filter(df.job_type.isNotNull())
df_unknown = df.filter(df.job_type.isNull())


# COMMAND ----------

# Step 5: Vectorize the features
from pyspark.ml.feature import VectorAssembler

assembler = VectorAssembler(
    inputCols=["age", "income"],
    outputCol="features")

df_known = assembler.transform(df_known)


# COMMAND ----------

# Step 6: Convert categorical labels to indices
from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="job_type", outputCol="label")
indexer_model = indexer.fit(df_known)
df_known = indexer_model.transform(df_known)

# COMMAND ----------

# Step 7: Train the Logistic Regression model
from pyspark.ml.classification import LogisticRegression

lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=10, regParam=0.3)
model = lr.fit(df_known)

# COMMAND ----------

# Step 8: Predict the missing job_type
df_unknown = assembler.transform(df_unknown)
predictions = model.transform(df_unknown)


# COMMAND ----------

# Step 9: Convert predicted indices back to original labels
from pyspark.ml.feature import IndexToString

converter = IndexToString(inputCol="prediction", outputCol="predictedJobType", labels=indexer_model.labels)
df_unknown = converter.transform(predictions)


# Display the predicted values
df_unknown.select("age", "income", "predictedJobType").show()


# COMMAND ----------


