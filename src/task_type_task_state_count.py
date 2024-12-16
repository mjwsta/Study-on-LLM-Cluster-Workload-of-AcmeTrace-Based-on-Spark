import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, sum
import os
os.environ['JAVA_HOME'] = "/opt/bitnami/java"  # 根据实际情况修改

# 初始化Spark Session
spark = SparkSession.builder \
    .appName("SerenTaskStatusAnalysis") \
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

# 计算每个任务类型（Eval 或 Pretrain）中，不同任务状态的占比
def compute_task_status_percentage(df, task_type):
    task_type_df = df.filter(col("TaskType") == task_type)
    
    # 计算每种任务状态的数量
    task_status_count = task_type_df.groupBy("TaskStatus").count()
    
    # 将任务状态列的值替换成相应标签
    task_status_count = task_status_count.withColumn("TaskStatus", 
                                                     when(col("TaskStatus") == "COMPLETED", "Completed")
                                                     .when(col("TaskStatus") == "CANCELED", "Canceled")
                                                     .when(col("TaskStatus") == "FAILED", "Failed")
                                                     .otherwise("Unknown"))
    
    # 获取每个任务状态的数量
    task_status_data = task_status_count.select("TaskStatus", "count").collect()
    
    # 提取每个状态的数量
    task_status_dict = {row["TaskStatus"]: row["count"] for row in task_status_data}
    
    # 计算每个任务状态的占比
    total_count = sum(task_status_dict.values())
    percentages = {status: (count / total_count) * 100 if total_count > 0 else 0
                  for status, count in task_status_dict.items()}
    
    return percentages

# 计算 Eval 和 Pretrain 任务类型中的任务状态占比
eval_task_status_percentages = compute_task_status_percentage(tasks_seren_df, "Eval")
pretrain_task_status_percentages = compute_task_status_percentage(tasks_seren_df, "Pretrain")

# 准备饼图数据
sizes_eval = [eval_task_status_percentages.get(status, 0) for status in ['Completed', 'Canceled', 'Failed']]
sizes_pretrain = [pretrain_task_status_percentages.get(status, 0) for status in ['Completed', 'Canceled', 'Failed']]

# 标签和颜色设置
labels = ['Completed', 'Canceled', 'Failed']
colors = ['#196b24', '#e97132', '#156082']

# 创建一个包含两个子图的图形
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Eval任务的饼图
axes[0].pie(sizes_eval, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
axes[0].set_title('Eval')

# Pretrain任务的饼图
axes[1].pie(sizes_pretrain, autopct='%1.1f%%', startangle=90, colors=colors, wedgeprops={'edgecolor': 'black'})
axes[1].set_title('Pretrain')

# 创建自定义图例句柄
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors]

# 放置图例在每个子图下方
fig.legend(handles, labels, loc='lower center', fontsize=12, ncol=3)

# 保存图形到指定路径
output_path = "results/1.3.png"
fig.savefig(output_path)

# 停止 Spark 会话
spark.stop()
