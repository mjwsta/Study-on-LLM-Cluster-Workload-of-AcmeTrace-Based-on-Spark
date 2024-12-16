import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when
import os
from py4j.java_gateway import java_import
os.environ['JAVA_HOME'] = "/opt/bitnami/java"  # 根据实际情况修改

# 初始化 Spark Session
spark = SparkSession.builder \
    .appName("SerenPretrainGpuAnalysis") \
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

# 过滤出 Pretrain 类型的任务
pretrain_tasks_seren_df = tasks_seren_df.filter(col("TaskType") == "Pretrain")

# 处理 GPU 使用个数区间和生成标签
gpu_bins = list(range(0, 2100, 150))
gpu_labels = [f'{gpu_bins[i]}-{gpu_bins[i+1]}' for i in range(len(gpu_bins)-1)]

# 将 GPU 使用个数区间分类
pretrain_tasks_seren_df = pretrain_tasks_seren_df.withColumn(
    "GpuUseCountBin",
    when((col("GpuUseCount") >= gpu_bins[0]) & (col("GpuUseCount") < gpu_bins[1]), gpu_labels[0])
    .when((col("GpuUseCount") >= gpu_bins[1]) & (col("GpuUseCount") < gpu_bins[2]), gpu_labels[1])
    .when((col("GpuUseCount") >= gpu_bins[2]) & (col("GpuUseCount") < gpu_bins[3]), gpu_labels[2])
    .when((col("GpuUseCount") >= gpu_bins[3]) & (col("GpuUseCount") < gpu_bins[4]), gpu_labels[3])
    .when((col("GpuUseCount") >= gpu_bins[4]) & (col("GpuUseCount") < gpu_bins[5]), gpu_labels[4])
    .when((col("GpuUseCount") >= gpu_bins[5]) & (col("GpuUseCount") < gpu_bins[6]), gpu_labels[5])
    .when((col("GpuUseCount") >= gpu_bins[6]) & (col("GpuUseCount") < gpu_bins[7]), gpu_labels[6])
    .when((col("GpuUseCount") >= gpu_bins[7]) & (col("GpuUseCount") < gpu_bins[8]), gpu_labels[7])
    .when((col("GpuUseCount") >= gpu_bins[8]) & (col("GpuUseCount") < gpu_bins[9]), gpu_labels[8])
    .when((col("GpuUseCount") >= gpu_bins[9]) & (col("GpuUseCount") < gpu_bins[10]), gpu_labels[9])
    .when((col("GpuUseCount") >= gpu_bins[10]) & (col("GpuUseCount") < gpu_bins[11]), gpu_labels[10])
    .when((col("GpuUseCount") >= gpu_bins[11]) & (col("GpuUseCount") < gpu_bins[12]), gpu_labels[11])
    .when((col("GpuUseCount") >= gpu_bins[12]) & (col("GpuUseCount") < gpu_bins[13]), gpu_labels[12])
    .when((col("GpuUseCount") >= gpu_bins[13]) & (col("GpuUseCount") <= gpu_bins[14]), gpu_labels[13])
    .otherwise("Unknown")
)

# 计算每个区间的平均排队时延
avg_queue_time_df = pretrain_tasks_seren_df.groupBy("GpuUseCountBin").avg("queue").orderBy("GpuUseCountBin")

# 收集数据
avg_queue_time_data = avg_queue_time_df.collect()

# 准备柱状图的数据
gpu_bins = [row["GpuUseCountBin"] for row in avg_queue_time_data]
avg_queue_times = [row["avg(queue)"] for row in avg_queue_time_data]

# 绘制柱状图
fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(gpu_bins, avg_queue_times, color='skyblue')

# 设置标题和标签
ax.set_xlabel("gpu use count", fontsize=12)
ax.set_ylabel("pretrain avg queue time(s)", fontsize=12)
ax.set_xticklabels(gpu_bins, rotation=45, ha='right')

# 显示图形
plt.tight_layout()

# 保存图像
output_path = "results/1.5.png"
fig.savefig(output_path)

# 停止 Spark 会话
spark.stop()
