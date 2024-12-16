from pyspark.sql import SparkSession
import os
os.environ['JAVA_HOME'] = "/opt/bitnami/java"

hostname = '77a50632bee4'
    # .master("local[*]") \

spark = SparkSession.builder \
    .appName("MyAPP") \
    .master(f"spark://{hostname}:7077") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "2") \
    .getOrCreate()

# hdfs path
hdfs_file_path = "hdfs://192.168.253.131:9000/AcmeTrace/data/job_trace/trace_kalos.csv"
df = spark.read.format('csv').option("header", "true").load(hdfs_file_path)
# df = spark.read.format("csv").option("header", "false").load(hdfs_file_path)
df.show()
# 
# df.printSchema()
# 停止 SparkSession 
spark.stop()