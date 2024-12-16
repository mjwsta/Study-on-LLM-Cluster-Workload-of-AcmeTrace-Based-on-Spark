from pyspark.sql import SparkSession
from pyspark.sql.functions import col, expr
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA, OneHotEncoder, StringIndexer, MinMaxScaler
from pyspark.ml import Pipeline
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
import os
from py4j.java_gateway import java_import
os.environ['JAVA_HOME'] = "/opt/bitnami/java"  # 根据实际情况修改


# 初始化Spark Session
spark = SparkSession.builder \
    .appName("Predictive_analysis") \
    .master("spark://{hostname}:7077") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", 10) \
    .getOrCreate()

# 1. 数据加载
hdfs_uri = "hdfs://192.168.253.131:9000"  # HDFS 地址
path = '/AcmeTrace/data/job_trace/trace_kalos.csv'
# print(url + path)
data = spark.read.format('csv').option("header", "true").load(hdfs_uri + path)
# 过滤状态不为 COMPLETED 的数据
data = data.filter(col("state") == "COMPLETED")

# 将必要的列转换为数值类型
data = data.withColumn("node_num", col("node_num").cast("double")) \
             .withColumn("gpu_num", col("gpu_num").cast("double")) \
             .withColumn("cpu_num", col("cpu_num").cast("double")) \
             .withColumn("submit_time", col("submit_time").cast("double")) \
             .withColumn("queue", col("queue").cast("double")) \
             .withColumn("duration", col("duration").cast("double"))

# 删除 numeric_features 列中包含空值的行
data = data.dropna(subset=['node_num', 'gpu_num', 'cpu_num', 'queue', 'duration'])

# def get_random_samples(df, k):
    # return df.sample(withReplacement=False, fraction=k/df.count(), seed=42)

# 目标变量和特征
numeric_features = ['node_num', 'gpu_num', 'cpu_num', 'queue']
categorical_features = ['job_id', 'user', 'type', 'state', 'queue']

# 添加新的标签列：duration + queue
data = data.withColumn("label", col("duration"))

# 2. 特征处理
# 数值特征向量化
numeric_assembler = VectorAssembler(inputCols=numeric_features, outputCol="numeric_features")

# 数值特征归一化
scaler = MinMaxScaler(inputCol="numeric_features", outputCol="normalized_numeric_features")

# 分类型特征编码
string_indexers = [
    StringIndexer(inputCol=col, outputCol=col + "_index", handleInvalid="keep") for col in categorical_features
]
one_hot_encoders = [OneHotEncoder(inputCol=col + "_index", outputCol=col + "_vec") for col in categorical_features]

# 合并所有特征
assembler = VectorAssembler(
    inputCols=["normalized_numeric_features"] + [col + "_vec" for col in categorical_features],
    outputCol="features"
)

# 3. PCA
pca = PCA(k=3, inputCol="features", outputCol="pca_features")

# 4. 模型
gbt = GBTRegressor(featuresCol="pca_features", labelCol="label", maxIter=50, maxDepth=10)

# 5. 创建 Pipeline
stages = []
stages += [numeric_assembler, scaler]
stages += string_indexers + one_hot_encoders
stages += [assembler, pca, gbt]
pipeline = Pipeline(stages=stages)

# 6. 数据分割
train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
# 7. 模型训练
model = pipeline.fit(train_data)

# 8. 模型评估
predictions = model.transform(test_data)

evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="rmse")
rmse = evaluator.evaluate(predictions)
mae_evaluator = RegressionEvaluator(labelCol="label", predictionCol="prediction", metricName="mae")
mae = mae_evaluator.evaluate(predictions)

print(f"模型评估结果:")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")

# 如果需要提取特征重要性
gbt_model = model.stages[-1]
if hasattr(gbt_model, "featureImportances"):
    feature_importances = gbt_model.featureImportances
    print("特征重要性:")
    print(feature_importances)
