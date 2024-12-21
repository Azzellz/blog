---
tags:
    - 编程
categories:
    - [编程, python]
title: Pandas库快速入门教程
---

下面是一个详细的 Python **pandas**快速教程，涵盖了从基础到常见数据操作的内容。可以直接复制使用！

---

### **1. 什么是 Pandas？**

Pandas 是 Python 中一个强大的数据处理与分析库，用于操作结构化数据（如表格、CSV 文件等）。它提供了两种核心数据结构：

-   **Series**: 一维数组（带索引）
-   **DataFrame**: 二维表格数据（类似 Excel 表）

安装 Pandas:

```bash
pip install pandas
```

---

### **2. 基础操作**

#### **导入 Pandas**

```python
import pandas as pd
```

#### **创建数据结构**

-   **Series (一维数据)**

```python
# 从列表创建
s = pd.Series([1, 2, 3, 4, 5])
print(s)

# 指定索引
s = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(s)
```

-   **DataFrame (二维数据)**

```python
# 从字典创建
data = {'Name': ['Alice', 'Bob', 'Charlie'],
        'Age': [25, 30, 35],
        'Salary': [50000, 60000, 70000]}
df = pd.DataFrame(data)
print(df)

# 从列表创建
data = [[1, 2], [3, 4], [5, 6]]
df = pd.DataFrame(data, columns=['Column1', 'Column2'])
print(df)
```

---

### **3. 数据导入与导出**

#### **读取数据**

-   **CSV 文件**

```python
df = pd.read_csv('data.csv')
```

-   **Excel 文件**

```python
df = pd.read_excel('data.xlsx')
```

-   **JSON 文件**

```python
df = pd.read_json('data.json')
```

#### **保存数据**

-   **保存为 CSV**

```python
df.to_csv('output.csv', index=False)
```

-   **保存为 Excel**

```python
df.to_excel('output.xlsx', index=False)
```

---

### **4. 数据探索**

#### **查看数据基本信息**

```python
print(df.head())    # 前5行
print(df.tail())    # 后5行
print(df.shape)     # 数据行列数
print(df.columns)   # 列名
print(df.info())    # 数据类型与非空信息
print(df.describe()) # 数值列的统计信息
```

#### **检查空值**

```python
print(df.isnull())          # 查看是否有空值
print(df.isnull().sum())    # 按列统计空值个数
```

---

### **5. 数据选择与过滤**

#### **选择列**

```python
print(df['Name'])          # 选择单列
print(df[['Name', 'Age']]) # 选择多列
```

#### **选择行**

-   **通过索引选择行**

```python
print(df.loc[0])   # 按标签选择
print(df.iloc[0])  # 按位置选择
```

-   **切片选择**

```python
print(df[1:3])  # 选择第2行到第3行（不包含第4行）
```

#### **条件过滤**

```python
print(df[df['Age'] > 30])  # 选择年龄大于30的行
```

---

### **6. 数据操作**

#### **新增列**

```python
df['Bonus'] = df['Salary'] * 0.1
print(df)
```

#### **修改列名**

```python
df.rename(columns={'Name': 'Employee Name'}, inplace=True)
print(df)
```

#### **删除列或行**

-   删除列：

```python
df.drop('Bonus', axis=1, inplace=True)
```

-   删除行：

```python
df.drop(0, axis=0, inplace=True)  # 删除第1行
```

#### **排序数据**

```python
# 按单列排序
df.sort_values('Age', ascending=False, inplace=True)

# 按多列排序
df.sort_values(['Age', 'Salary'], ascending=[True, False], inplace=True)
```

---

### **7. 数据清洗**

#### **处理缺失值**

-   **填充空值**

```python
df['Age'].fillna(df['Age'].mean(), inplace=True)  # 用平均值填充
df.fillna(0, inplace=True)  # 用0填充所有空值
```

-   **删除空值**

```python
df.dropna(inplace=True)  # 删除包含空值的行
```

#### **重复值处理**

```python
df.drop_duplicates(inplace=True)
```

---

### **8. 数据分组与聚合**

#### **分组操作**

```python
grouped = df.groupby('Department')
print(grouped['Salary'].mean())  # 按部门计算平均工资
```

#### **聚合操作**

```python
print(df.groupby('Department').agg({'Salary': ['mean', 'sum']}))
```

---

### **9. 合并与连接**

#### **合并两个 DataFrame**

```python
df1 = pd.DataFrame({'ID': [1, 2], 'Name': ['Alice', 'Bob']})
df2 = pd.DataFrame({'ID': [1, 2], 'Salary': [50000, 60000]})

# 按列合并
merged = pd.merge(df1, df2, on='ID')
print(merged)
```

#### **追加数据**

```python
df3 = pd.DataFrame({'ID': [3], 'Name': ['Charlie'], 'Salary': [70000]})
df = pd.concat([df, df3], ignore_index=True)
print(df)
```

---

### **10. 时间序列操作**

Pandas 支持对时间序列数据的处理。

```python
# 创建时间序列
date_range = pd.date_range(start='2023-01-01', end='2023-01-07', freq='D')
print(date_range)

# 设置为索引
df['Date'] = date_range[:len(df)]
df.set_index('Date', inplace=True)
```

---

### **11. 数据可视化**

可以结合 Matplotlib 或 Seaborn 使用：

```python
import matplotlib.pyplot as plt

# 简单绘图
df['Salary'].plot(kind='bar')
plt.show()
```

---

### **12. 导出和保存总结**

完成数据分析后，保存处理后的数据为不同文件格式，常用的方法如：

```python
df.to_csv('final_output.csv', index=False)
df.to_excel('final_output.xlsx', index=False)
```

---

以上内容涵盖了 Pandas 的主要功能，适合快速入门和日常数据分析需求。如果需要更深层次的操作，建议查阅 [Pandas 官方文档](https://pandas.pydata.org/docs/)！
