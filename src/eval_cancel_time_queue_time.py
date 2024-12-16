import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import os
os.environ['JAVA_HOME'] = "/opt/bitnami/java"  # 根据实际情况修改

# 初始化 Spark Session
spark = SparkSession.builder \
    .appName("SerenEvalCancelQueueTimeAnalysis") \
    .master("spark://localhost:7077") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", 10) \
    .getOrCreate()

hdfs_uri = "hdfs://192.168.253.131:9000"  # HDFS 地址

# 表路径
table_paths = {
    "tasks_seren": "/AcmeTrace/data/job_trace/seren_tasks.csv",  # Seren 集群任务
}

# 加载任务表
def load_table(path):
    df = spark.read.format('csv').option("header", "true").load(hdfs_uri + path)
    return df

# 加载 Seren 集群的任务数据
tasks_seren_df = load_table(table_paths["tasks_seren"])

# 过滤出 Eval 类型的任务
eval_tasks_seren_df = tasks_seren_df.filter(col("TaskType") == "Eval")

# 处理排队时延区间
queue_bins = list(range(0, 301, 20))  # 排队时延区间：0~20, 20~40, ..., 260~280, >300
queue_labels = [f'{queue_bins[i]}-{queue_bins[i+1]}' for i in range(len(queue_bins)-1)] + [">300"]

# 为排队时延分配区间
eval_tasks_seren_df = eval_tasks_seren_df.withColumn(
    "QueueTimeBin",
    when((col("queue") >= queue_bins[0]) & (col("queue") < queue_bins[1]), queue_labels[0])
    .when((col("queue") >= queue_bins[1]) & (col("queue") < queue_bins[2]), queue_labels[1])
    .when((col("queue") >= queue_bins[2]) & (col("queue") < queue_bins[3]), queue_labels[2])
    .when((col("queue") >= queue_bins[3]) & (col("queue") < queue_bins[4]), queue_labels[3])
    .when((col("queue") >= queue_bins[4]) & (col("queue") < queue_bins[5]), queue_labels[4])
    .when((col("queue") >= queue_bins[5]) & (col("queue") < queue_bins[6]), queue_labels[5])
    .when((col("queue") >= queue_bins[6]) & (col("queue") < queue_bins[7]), queue_labels[6])
    .when((col("queue") >= queue_bins[7]) & (col("queue") < queue_bins[8]), queue_labels[7])
    .when((col("queue") >= queue_bins[8]) & (col("queue") < queue_bins[9]), queue_labels[8])
    .when((col("queue") >= queue_bins[9]) & (col("queue") < queue_bins[10]), queue_labels[9])
    .when((col("queue") >= queue_bins[10]) & (col("queue") <= 300), queue_labels[10])
    .otherwise(">300")
)

# 计算每个排队时延区间内的任务取消次数
cancel_times_df = eval_tasks_seren_df.filter(col("Status") == "CANCELED") \
    .groupBy("QueueTimeBin").count().orderBy("QueueTimeBin")

# 收集数据
cancel_times_data = cancel_times_df.collect()

# 准备柱状图的数据
queue_time_bins = [row["QueueTimeBin"] for row in cancel_times_data]
cancel_times = [row["count"] for row in cancel_times_data]

# 绘制柱状图
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(queue_time_bins, cancel_times, color='lightcoral')

# 设置标签
ax.set_xlabel("queue time (s)", fontsize=12)
ax.set_ylabel("eval cancel times", fontsize=12)
ax.set_xticklabels(queue_time_bins, rotation=45, ha='right')

# 设置纵轴标签
ax.set_yticks(range(0, max(cancel_times) + 2000, 2000))

# 显示图形
plt.tight_layout()

# 保存图像
output_path = "results/1.7.png"
fig.savefig(output_path)

# 停止 Spark 会话
spark.stop()
