import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum
import os
os.environ['JAVA_HOME'] = "/opt/bitnami/java"  # 根据实际情况修改

# 初始化Spark Session
spark = SparkSession.builder \
    .appName("TaskGPUTimeAnalysis") \
    .master("spark://localhost:7077") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", 10) \
    .getOrCreate()

hdfs_uri = "hdfs://192.168.253.131:9000"  # HDFS 地址

# 表路径
table_paths = {
    "tasks_seren": "/AcmeTrace/data/job_trace/seren_tasks.csv",  # Seren 集群任务
    "tasks_kalos": "/AcmeTrace/data/job_trace/kalos_tasks.csv"   # Kalos 集群任务
}

# 加载任务表
def load_table(path):
    df = spark.read.format('csv').option("header", "true").load(hdfs_uri + path)
    return df

# 加载 Seren 和 Kalos 集群的任务数据
tasks_seren_df = load_table(table_paths["tasks_seren"])
tasks_kalos_df = load_table(table_paths["tasks_kalos"])

# 计算每个集群中，不同任务状态的 GPU 时间占比
def compute_gpu_time_percentage(df, cluster_name):
    task_status_gpu_time = df.groupBy("TaskStatus").agg(sum("GPU_Time").alias("Total_GPU_Time"))
    
    # 将任务状态列的值替换成相应标签
    task_status_gpu_time = task_status_gpu_time.withColumn("TaskStatus", 
                                                           when(col("TaskStatus") == "COMPLETED", "Completed")
                                                           .when(col("TaskStatus") == "CANCELED", "Canceled")
                                                           .when(col("TaskStatus") == "FAILED", "Failed")
                                                           .otherwise("Unknown"))
    
    # 获取每个任务状态的 GPU 时间
    gpu_time_data = task_status_gpu_time.select("TaskStatus", "Total_GPU_Time").collect()
    
    # 提取每个状态的 GPU 时间
    gpu_time_dict = {row["TaskStatus"]: row["Total_GPU_Time"] for row in gpu_time_data}
    
    # 计算每个状态的 GPU 时间占比（百分比）
    total_gpu_time = sum(gpu_time_dict.values())
    percentages = {status: (gpu_time / total_gpu_time) * 100 if total_gpu_time > 0 else 0
                  for status, gpu_time in gpu_time_dict.items()}
    
    return percentages

# 计算 Seren 和 Kalos 集群的任务状态 GPU 时间占比
seren_gpu_percentages = compute_gpu_time_percentage(tasks_seren_df, "Seren")
kalos_gpu_percentages = compute_gpu_time_percentage(tasks_kalos_df, "Kalos")

# 准备饼图数据
sizes_seren = [seren_gpu_percentages.get(status, 0) for status in ['Completed', 'Canceled', 'Failed']]
sizes_kalos = [kalos_gpu_percentages.get(status, 0) for status in ['Completed', 'Canceled', 'Failed']]

# 标签和颜色设置
labels = ['Completed', 'Canceled', 'Failed']
colors = ['#4CAF50', '#FF6347', '#FFA500']  # 完成 - 绿色, 取消 - 红色, 失败 - 橙色

# 创建一个包含两个子图的图形
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Seren的饼图
axes[0].pie(sizes_seren, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
axes[0].set_title('Seren')

# Kalos的饼图
axes[1].pie(sizes_kalos, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
axes[1].set_title('Kalos')

# 创建自定义图例句柄
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]

# 合并图例，放置在右上角
fig.legend(handles, labels, loc='upper right', fontsize=12)

# 使用 tight_layout 以确保布局不会被挤出
plt.subplots_adjust(top=0.85)  # 留出更多上方的边距

# 保存图形到指定路径
output_path = "results/1.2.png"
fig.savefig(output_path)

# 停止 Spark 会话
spark.stop()
