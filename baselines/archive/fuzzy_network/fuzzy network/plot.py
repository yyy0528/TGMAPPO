import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
import numpy as np

plt.rcParams['font.family'] = 'Times New Roman'

# 1. 读取数据
with open("./result/dr.txt", "r") as f:
    lines_1 = f.readlines()

with open("./result/dr_mean_obs.txt", "r") as f:
    lines_2 = f.readlines()

font_prop = FontProperties(family='Times New Roman')
# 求中位数
def find_median_for_each_sublist(lst, sublist_size):
    medians = []
    for i in range(0, len(lst), sublist_size):
        sublist = lst[i:i+sublist_size]
        sublist.sort()
        n = len(sublist)
        if n % 2 == 1:
            median = sublist[n // 2]
        else:
            middle1 = sublist[n // 2 - 1]
            middle2 = sublist[n // 2]
            median = (middle1 + middle2) / 2
        medians.append(median)
    return medians

length = 1000
# 2. 处理数据
values_1 = []
k = 0
tmp_last = 0
for line in lines_1:
    k += 1
    if k > length:
        break
    try:
        tmp = float(line.strip())
        # if tmp_last - tmp > 500 or tmp - tmp_last > 500:
        #     continue
        tmp_last = tmp

        values_1.append(tmp)
    except ValueError:
        print(f"Invalid value in file: {line.strip()}")
values_2 = []
k = 0
for line in lines_2:
    k += 1
    if k > length:
        break
    try:

        tmp = float(line.strip())


        values_2.append(tmp)
    except ValueError:
        print(f"Invalid value in file: {line.strip()}")

# 示例
 # 创建一个包含3000个连续整数的列表
sublist_size = 10  # 每个子列表的大小
medians_1 = find_median_for_each_sublist(values_1, sublist_size)
medians_2 = find_median_for_each_sublist(values_2, sublist_size)
# new_array = []
# for value in medians:
#     for _ in range(10):
#         new_array.append(value)

averages = []

# for i in range(0, len(values_1), 25):
#     group_data = values_1[i:i+25]  # 获取列表中连续的10个元素
#     group_average = sum(group_data) / len(group_data)  # 计算平均值
#     averages.append(group_average)

x_m = []
for i in range(len(medians_1)):
    x_m.append((i+1)*10)

# print(x_m)




# 使用线性插值函数进行插值


# 3. 绘制图像
plt.figure(figsize=(6, 5), constrained_layout=True)
# plt.plot(values_1, c='wheat',alpha=0.3)
# # plt.plot(x_m, averages, label='back', c='')
#
# plt.plot(values_2, c='lightblue',alpha=0.3)
# plt.plot(values_3, c='mistyrose',alpha=0.3)
# plt.plot(values_1, c='orange',alpha=0.2)

plt.plot(values_1, c='red',alpha=0.2)
plt.plot(values_2, c='indigo',alpha=0.2)


# 绘制第二条曲线
plt.plot(x_m, medians_1, label='proposed', c='red', zorder=5)
plt.plot(x_m, medians_2, label='MASAC', c='m', zorder=4)


plt.xlabel('episode', fontsize=20)
plt.ylabel('return', fontsize=20)

plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
# 显示网格
plt.grid(True)
plt.grid(True)
plt.legend(fontsize=16)
plt.show()
# plt.savefig('new_delay_16.png', dpi=150)