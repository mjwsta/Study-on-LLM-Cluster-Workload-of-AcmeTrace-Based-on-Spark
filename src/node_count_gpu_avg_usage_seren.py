import matplotlib.pyplot as plt
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, avg
import pandas as pd
import os
os.environ['JAVA_HOME'] = "/opt/bitnami/java"  # 根据实际情况修改

# 初始化 Spark Session
spark = SparkSession.builder \
    .appName("SerenGPUUsageHistogram") \
    .master("spark://localhost:7077") \
    .config("spark.executor.memory", "4g") \
    .config("spark.driver.memory", "4g") \
    .config("spark.sql.shuffle.partitions", 10) \
    .getOrCreate()

hdfs_uri = "hdfs://192.168.253.131:9000"  # HDFS 地址

# 表路径
table_paths = {
    "gpu_util_seren": "/AcmeTrace/data/utilization/seren/GPU_UTIL.csv",  # Seren 集群的 GPU 利用率
}

# 加载表格并清洗列名（替换为自增数字，保留Time列）
def load_table(path):
    """
    读取指定路径的数据表，并清洗列名（替换为自增数字）
    """
    df = spark.read.format('csv').option("header", "true").load(hdfs_uri + path)
    
    # 获取原始列名
    columns = df.columns
    
    # 保留 "Time" 列，其他列名替换为自增数字
    new_columns = ["Time"] + [f"col_{i}" for i in range(1, len(columns))]
    
    # 重新设置列名
    df = df.toDF(*new_columns)
    
    return df

# 加载 Seren 集群 GPU 利用率数据
gpu_util_seren_df = load_table(table_paths["gpu_util_seren"])

# 获取数值列（不考虑时间列）
def get_numeric_columns(df):
    """
    获取数据框中的数值列（除去时间列）
    """
    numeric_columns = [col for col in df.columns if col != "Time"]
    return numeric_columns

# 获取 GPU 利用率的数值列
gpu_columns_seren = get_numeric_columns(gpu_util_seren_df)

# 计算 GPU 使用情况（每个节点的平均使用率）
gpu_avg_usage_seren_df = gpu_util_seren_df.select(*gpu_columns_seren).agg(*[avg(col(c)).alias(c) for c in gpu_columns_seren])

# 转换为 Pandas DataFrame 用于图表展示
gpu_avg_usage_seren_pdf = gpu_avg_usage_seren_df.toPandas()

# 获取所有节点的平均 GPU 使用率
usage_data = gpu_avg_usage_seren_pdf.mean(axis=0)

# 创建直方图，分为 10 个区间
plt.figure(figsize=(10, 6))
n, bins, patches = plt.hist(usage_data, bins=10, edgecolor="black", alpha=0.75)

# 在每个柱子上显示数值
for i in range(len(n)):
    x = (bins[i] + bins[i + 1]) / 2  # 每个柱子的中心
    y = n[i]  # 每个柱子的高度
    plt.text(x, y + 2, str(int(y)), ha="center", fontsize=10)  # 数值显示在柱上方

# 设置标题和标签
plt.title("Histogram of Node Average GPU Usage", fontsize=14)
plt.xlabel("Average GPU Usage (%)", fontsize=12)
plt.ylabel("Number of Nodes", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)

# 保存图表为文件
output_image_path = "results/1.8.png"
plt.savefig(output_image_path)
print(f"直方图已保存到 {output_image_path}")

# 停止 Spark 会话
spark.stop()
