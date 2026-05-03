
import os


import pandas as pd


import numpy as np


import matplotlib.pyplot as plt


from sklearn.model_selection import train_test_split

# 导入随机森林回归模型，用于做订单量预测
from sklearn.ensemble import RandomForestRegressor

# 导入模型评估指标，计算预测误差（MAE平均绝对误差、RMSE均方根误差）
from sklearn.metrics import mean_absolute_error, mean_squared_error


# 导入警告管理库，用于关闭不必要的系统提示
import warnings

# 关闭所有警告信息，让控制台输出干净整洁
warnings.filterwarnings("ignore")

# 设置matplotlib使用默认英文字体，彻底避免中文乱码/字体警告
plt.rcParams["font.family"] = ["DejaVu Sans"]

# 设置负号正常显示，防止图表中负号变成方框
plt.rcParams["axes.unicode_minus"] = False

# 自动创建 outputs 文件夹，用于保存生成的图表
# exist_ok=True 表示如果文件夹已存在则不报错
os.makedirs("outputs", exist_ok=True)

# M1 数据处理模块
def m1_data_process():
    """
    功能：加载出租车数据，进行数据清洗与特征提取
    步骤：读取数据 → 删除缺失值 → 过滤异常值 → 提取时间特征
    """
    # 打印模块开始提示
    print(" Data Processing Start ")

    # 从指定绝对路径读取 parquet 格式的出租车数据
    # r 表示原生字符串，防止路径中的反斜杠被转义
    df = pd.read_parquet(r"D:\1\taxi\data\yellow_tripdata_2023-01.parquet")

    # 删除数据中包含缺失值的行，保证数据完整性
    df = df.dropna()

    # 过滤异常行程距离：只保留距离大于0且小于100英里的记录
    df = df[(df["trip_distance"] > 0) & (df["trip_distance"] < 100)]

    # 过滤异常车费：只保留费用大于0且小于500美元的记录
    df = df[(df["fare_amount"] > 0) & (df["fare_amount"] < 500)]

    # 将上车时间字符串转换为标准日期时间格式，方便后续提取时间特征
    df["tpep_pickup_datetime"] = pd.to_datetime(df["tpep_pickup_datetime"])

    # 从上车时间中提取小时（0-23），作为后续分析与预测的关键特征
    df["hour"] = df["tpep_pickup_datetime"].dt.hour

    # 打印模块完成提示
    print(" Data Processing Done ")

    # 返回清洗完成后的数据集，供后续模块使用
    return df

# M2 数据可视化模块
def m2_visualize(df):
    """
    功能：根据清洗后的数据生成两张分析图表
    图表1：每小时打车需求量折线图
    图表2：热门上车区域TOP10柱状图
    """
    # 打印模块开始提示
    print(" Generating Charts ")
    
    # figsize=(10,5) 设置图片大小为宽10、高5
    df.groupby("hour").size().plot(kind="line", figsize=(10, 5))

    # 设置图表标题
    plt.title("Hourly Taxi Demand")

    # 设置X轴标签（小时）
    plt.xlabel("Hour")

    # 设置Y轴标签（订单数量）
    plt.ylabel("Number of Trips")

    # 将图表保存到 outputs 文件夹下
    plt.savefig("outputs/hourly_demand.png")

    # 关闭当前图表，释放内存，防止图表重叠
    plt.close()

    
    # 统计上车区域ID出现次数，取前10名，绘制柱状图
    df["PULocationID"].value_counts().head(10).plot(kind="bar", figsize=(10, 5))

    # 设置图表标题
    plt.title("Top 10 Pickup Zones")

    # 设置X轴标签（区域编号）
    plt.xlabel("Location ID")

    # 设置Y轴标签（订单数量）
    plt.ylabel("Trips")

    # 保存图表
    plt.savefig("outputs/top_zones.png")

    # 关闭图表
    plt.close()
    
    # 打印保存完成提示
    print(" Charts Saved to outputs/ ")

#  M3 预测模型模块 
def m3_model(df):
    """
    功能：构建随机森林模型，根据【区域ID + 小时】预测订单需求量
    """
    # 打印模块开始提示
    print("Training Prediction Model ")

    # 按区域ID和小时分组，统计每个组合的订单数，命名为 demand
    df_agg = df.groupby(["PULocationID", "hour"]).size().reset_index(name="demand")

    # 构建特征 X：区域ID 和 小时（用来预测的输入）
    X = df_agg[["PULocationID", "hour"]]

    # 构建标签 y：订单需求量（需要预测的目标）
    y = df_agg["demand"]
    
    # 将数据按 8:2 划分
    # 80% 用于训练模型，20% 用于测试模型效果
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # 创建随机森林回归模型，设置30棵树（保证速度与精度平衡）
    model = RandomForestRegressor(n_estimators=30)

    # 使用训练集数据训练模型
    model.fit(X_train, y_train)
    
    # 使用训练好的模型对测试集进行预测
    pred = model.predict(X_test)

    # 计算并输出平均绝对误差（MAE），越小表示预测越准
    print(f"MAE: {mean_absolute_error(y_test, pred):.2f}")

    # 计算并输出均方根误差（RMSE），越小表示预测越准
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")

    # 打印模型训练完成提示
    print(" Model Training Done ")

#  M4 智能问答系统模块 
def m4_qa(df):
    """
    功能：命令行交互式问答系统
    支持查询：高峰时段、热门区域、平均车费、退出程序
    """
    
    print("\n Final Assignment QA System ")
    print("Commands: peak, zone, fare, exit\n")
    
    # 无限循环，实现持续问答，直到输入 exit 退出
    while True:
        # 获取用户输入的问题
        q = input("Enter question: ")

        # 如果用户输入 exit，跳出循环，结束问答
        if q == "exit":
            break

        # 如果输入包含 peak，查询并输出打车高峰小时
        elif "peak" in q:
            h = df["hour"].value_counts().idxmax()
            print(f"Peak hour: {h}")

        # 如果输入包含 zone，查询并输出热门区域TOP5
        elif "zone" in q:
            print("Top zones:\n", df["PULocationID"].value_counts().head())

        # 如果输入包含 fare，查询并输出平均车费
        elif "fare" in q:
            avg = df["fare_amount"].mean()
            print(f"Average fare: {avg:.2f}")

        # 输入不支持的指令，给出提示
        else:
            print("Try: peak, zone, fare, exit")

#  主程序入口
# 当直接运行此文件时，执行以下逻辑
if __name__ == "__main__":
    # 1. 执行数据处理，得到清洗后的数据
    df = m1_data_process()

    # 2. 执行可视化，生成两张图表
    m2_visualize(df)

    # 3. 训练预测模型并输出评估结果
    m3_model(df)

    # 4. 启动智能问答系统
    m4_qa(df)

    # 全部任务完成提示
    print("\n Assignment Completed!")
