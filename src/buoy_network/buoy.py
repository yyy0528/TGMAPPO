import datetime
import logging
import math
import os
import random
import numpy as np
import logging
from datetime import datetime
import os
import time
NUM = 100
# import logging
#
# logger = logging.getLogger("buoy_net")
import time
import logging
from datetime import datetime
import os
from .utils import setup_logger

logger = setup_logger(log_dir="outputs/TG-MAPPO/diagnostic_logs", log_name="buoy_2018")
class buoy_node:
    def __init__(self,grid_size):
        self.id = None
        self.energy = None
        self.o_energy = None
        self.position = None
        self.neighbor = []  # 这个也设置成列表吧
        self.neighbor_two = []
        self.packet = 0
        self.if_sink = False
        self.packet_buffer = None  # 考虑拥塞的情况才会用到
        self.lost_rate_list = []
        self.lost_rate = 0
        self.success_rate_neighbor = []
        self.success_number = []
        self.send_number = []
        self.lost_idx = 0
        self.max_speed = 0.1  # 最大移动速度
        self.accl = 0.001   # 加速度
        self.grid_size =grid_size
        # 新增：场景边界范围（默认0-grid_size的坐标范围，可根据实际场景调整）
        self.boundary = [0, self.grid_size]  # [min_bound, max_bound] 适用于x和y轴
        self.position_init = None
        self.sink_neighbor = []
    def initialize(self, buoy_id, energy, position):
        # print(f"initial,position:{position}")
        self.id = buoy_id
        self.energy = energy
        self.o_energy = energy
        self.position = position
        self.position_init = [0,0]
        # self.sink_neighbor = []
        # self._clamp_position()

        for i in range(10):
            random_int = random.randint(1, 8)
            self.lost_rate_list.append(random_int * 0.1)
            # self.lost_rate_list.append(0)
        self.lost_rate = self.lost_rate_list[0]
        # self.lost_rate =  0
        self.lost_idx += 1
        self.lost_rate = 0


        self.position_init[0] = position[0]
        self.position_init[1] = position[1]

    def get_info(self):
        info = {'id': self.id, 'energy': self.energy, 'position': self.position}
        return info

    def note_transmit(self):
        pass

    def send(self):  # 接受和发送还是要分开写，WSN中发送信息也是广播，所以函数里面应该不用考虑谁接收的问题
        self.packet -= 1
        # self.energy -= 0.01

    # 这个还可以用来统计网络的吞吐量
    def receive(self):

        rate = random.random()
        if rate >= self.lost_rate:
            self.packet += 1
            # self.energy -= 0.01
            return True
        else:
            return False

    def get_position(self):
        return self.position

    def set_neighbor(self, nb_id):
        self.neighbor.append(nb_id)

    def get_neighbor(self):
        return self.neighbor

    def set_neighbor_two(self, nb_id):
        self.neighbor_two.append(nb_id)

    def get_neighbor_two(self):
        return self.neighbor_two

    def node_refresh(self):
        # self.energy = self.o_energy
        self.packet = 0
        self.lost_rate = self.lost_rate_list[0]
        self.lost_idx = 1
        # 这个还得改
        # 新增：位置边界约束辅助方法
        self.lost_rate = 0

    def _clamp_position(self):
        """确保位置坐标在边界范围内"""
        min_bound, max_bound = self.boundary
        # 约束x坐标
        if self.position[0] < min_bound:
            self.position[0] = min_bound
        elif self.position[0] > max_bound:
            self.position[0] = max_bound
        # 约束y坐标
        if self.position[1] < min_bound:
            self.position[1] = min_bound
        elif self.position[1] > max_bound:
            self.position[1] = max_bound



    def move(self,):
        # 随机方向移动（示例：基于正态分布的随机游走）
        # dx = random.normalvariate(0, self.max_speed)
        dx = self.max_speed*(random.choice([1, 0, -1]))
        # dy = random.normalvariate(0, self.max_speed)
        dy = self.max_speed * (random.choice([1, 0, -1]))
        # 限制速度不超过最大值
        # print(f"dx:{dx},dy:{dy}")
        # print(f"pos_init:{self.position_init[0]},{self.position_init[1]}")

        # 更新位置
        if (self.position[0]+dx >=self.position_init[0]+0.1) :
            self.position[0] = self.position_init[0]+0.1
        elif (self.position[0]+dx <=self.position_init[0]-0.1) :
            self.position[0] = self.position_init[0] - 0.1
        else:
            self.position[0] = self.position[0] + dx


        if (self.position[1]+dy >=self.position_init[1]+0.1) :
            self.position[1] = self.position_init[1]+0.1
        elif (self.position[1]+dy <=self.position_init[1]-0.1) :
            self.position[1] = self.position_init[1] - 0.1
        else:
            self.position[1] = self.position[1] + dy



        if self.if_sink:
            self.position[0] = self.position_init[0]
            self.position[1] = self.position_init[1]

        if self.id in self.sink_neighbor:
            self.position[0] = self.position_init[0]
            self.position[1] = self.position_init[1]


        # # 临时计算新位置（先不更新）
        # new_x = self.position[0] + dx
        # new_y = self.position[1] + dy
        #
        # # 保存当前位置，用于碰撞检测回退
        # current_x, current_y = self.position[0], self.position[1]
        #
        # # 尝试更新位置
        # self.position[0] = new_x
        # self.position[1] = new_y
        #
        # # 应用边界约束
        # self._clamp_position()


class data_packet:
    def __init__(self, packet_id, source, destination):
        self.id = packet_id
        self.agent_id = source  # 当前位置
        self.last_pos = source  # 判断是否被传输
        self.stay_time = 1  # 停留时间
        self.memories = []
        self.route = []
        self.life = 80
        self.route.append(source)
        self.destination = destination
        self.if_sink = False
        self.change_n = 0
        self.q_time = [] # 排队时间


    def transmit(self, next_hop):
        self.route.append(next_hop)
        self.q_time.append(self.stay_time)
        self.agent_id = next_hop
        if next_hop == self.destination:
            self.if_sink = True
            self.q_time.append(0)
            # self.q_time.append(0)
        else:
            self.if_sink = False

    def add_memory(self, memory_id):
        self.memories.append(memory_id)


class buoy_net:
    def __init__(self, buoys,all_args):
        self.buoys = []
        self.count_NB = []  # 这个还没用上
        self.buoys_number = len(buoys)
        random.seed(2568)
        self.initialize_node(buoys,all_args)
        # 初始化局部网络
        self.adj_matrix = self.build_adjacency_matrix()
        self.adj_matrix_two = self.build_adjacency_matrix_two()
        self.initialize_neighbor()
        self.initialize_neighbor_two()
        self.added_value = np.zeros((len(self.buoys), len(self.buoys)), dtype=np.float32)
        self.build_s_rate(self.added_value)
        self.grid_size =all_args.grid_size
        # # 新增：日志配置
        # self.log_dir = "training_logs"
        # os.makedirs(self.log_dir, exist_ok=True)  # 创建日志目录
        # log_file = os.path.join(self.log_dir, f"log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        #
        # # 配置日志：同时输出到文件和控制台
        # logging.basicConfig(
        #     level=logging.INFO,
        #     format="%(asctime)s - %(levelname)s - %(message)s",
        #     handlers=[
        #         logging.FileHandler(log_file),
        #         logging.StreamHandler()
        #     ]
        # )
        # self.logger = logging.getLogger(__name__)



    def initialize_node(self, buoys,all_args):
        for i in range(self.buoys_number):
            buoy = buoy_node(all_args.grid_size)
            buoy.initialize(i, buoys[i][0], buoys[i][1])
            self.buoys.append(buoy)

    # 一跳邻居
    def build_adjacency_matrix(self):

        adj_matrix = np.zeros((self.buoys_number, self.buoys_number))
        # 只要通信范围内的节点的距离，这样通过列的序号就得到这个邻居的编号。
        for i in range(self.buoys_number):
            cc = 0  # 计数
            for j in range(self.buoys_number):
                dist = math.sqrt(
                    math.pow((self.buoys[i].get_position()[0] - self.buoys[j].get_position()[0]), 2) + math.pow(
                        (self.buoys[i].get_position()[1] - self.buoys[j].get_position()[1]), 2))
                # 欧式距离
                # + math.pow(
                #                     (self.buoys[i].get_position()[2] - self.buoys[j].get_position()[2]), 2)
                if dist <= 1.5:
                    adj_matrix[i][j] = dist  # 只要通信范围内的节点的距离，这样通过列的序号就得到这个邻居的编号。
                    cc += 1
                else:
                    adj_matrix[i][j] = 0
            self.count_NB.append(cc - 1)  # 统计邻居数量，减去把自己当邻居的情况
        '''if(math.sqrt(math.pow((x[i]-250),2)+math.pow((y[i]-250),2)+math.pow(z[i]-250,2)))<=self.r:
                candidate_of_cluster_heads.append(i) '''
        # if np.array(self.count_NB).all() >= 1:  # 每个节点的邻居数至少为2时，则终止循环，为了避免后续因邻居过少而报错
        #     self.do = False
        return adj_matrix

    # 两跳邻居
    def build_adjacency_matrix_two(self):

        adj_matrix = np.zeros((self.buoys_number, self.buoys_number))
        # 只要通信范围内的节点的距离，这样通过列的序号就得到这个邻居的编号。
        for i in range(self.buoys_number):
            cc = 0  # 计数
            for j in range(self.buoys_number):
                dist = math.sqrt(
                    math.pow((self.buoys[i].get_position()[0] - self.buoys[j].get_position()[0]), 2) + math.pow(
                        (self.buoys[i].get_position()[1] - self.buoys[j].get_position()[1]), 2))
                # 欧式距离
                # + math.pow(
                #                     (self.buoys[i].get_position()[2] - self.buoys[j].get_position()[2]), 2)
                if dist <= 3:
                    adj_matrix[i][j] = dist  # 只要通信范围内的节点的距离，这样通过列的序号就得到这个邻居的编号。
                    cc += 1
                else:
                    adj_matrix[i][j] = 0
            self.count_NB.append(cc - 1)  # 统计邻居数量，减去把自己当邻居的情况
        '''if(math.sqrt(math.pow((x[i]-250),2)+math.pow((y[i]-250),2)+math.pow(z[i]-250,2)))<=self.r:
                candidate_of_cluster_heads.append(i) '''
        # if np.array(self.count_NB).all() >= 1:  # 每个节点的邻居数至少为2时，则终止循环，为了避免后续因邻居过少而报错
        #     self.do = False
        return adj_matrix

    def initialize_neighbor(self):
        for buoy in self.buoys:
            for i in range(self.buoys_number):
                if self.adj_matrix[buoy.id][i] > 0:
                    buoy.set_neighbor(i)

            buoy.success_number = [0] * len(buoy.neighbor)
            buoy.send_number = [0] * len(buoy.neighbor)

    def initialize_neighbor_two(self):
        for buoy in self.buoys:
            for i in range(self.buoys_number):
                if self.adj_matrix_two[buoy.id][i] > 0:
                    buoy.set_neighbor_two(i)

    def get_buoys(self):
        return self.buoys

    def build_s_rate(self,added_value):
        print("buoy.success_rate_neighbor:\n")
        for buoy in self.buoys:

            buoy.success_rate_neighbor = []

            for action_id in range(self.buoys_number):

                if action_id in buoy.neighbor:
                    neighbor_id = action_id
                    success = 0
                    for i in range(10):
                        rate = random.random()
                        if rate >= self.buoys[neighbor_id].lost_rate:
                            success += 1
                    rate = success / 10
                    rate = 1        #成功率是1
                    buoy.success_rate_neighbor.append(rate)

                else:
                    buoy.success_rate_neighbor.append(0)


            for  i in buoy.success_rate_neighbor:
                print(i ,end="")
            print("  ",end=" ")
            for  j in range(len(added_value[buoy.id])):
                # print(int(added_value[buoy.id][j]),end=" ")
                # # 假设 added_value[buoy.id][j] 是一个浮点数，例如 12.3456
                # value = added_value[buoy.id][j]
                #
                # # 格式化输出：保留三位有效数字，宽度为 5（不足用空格补全）
                # formatted = f"{value:.3g}"  # .3g 表示保留三位有效数字
                # padded = formatted.ljust(3)  # 左对齐，用空格补全到宽度 5
                #
                # print(padded, end=" ")
                # 取整数部分（原代码中的 int 转换）
                num = int(added_value[buoy.id][j])

                # 格式化：总宽度3，右对齐（数字靠右，左侧补空格）
                print(f"{num:3d}", end=" ")



            print(" ")

            # for buoy in self.buoys:
            #     # 从列表中提取字符串（假设列表只有一个元素，取索引0）
            #     # 若列表有多个元素，需根据实际情况调整（如用join拼接）
            #     success_str = str(buoy.success_rate_neighbor[0])  # 提取列表中的字符串
            #
            #     # 左矩阵：字符串固定宽度（25字符，左对齐，不足补空格）
            #     left_row = success_str.ljust(25)  # 现在是字符串调用ljust，不会报错
            #
            #     # 处理右矩阵：数字格式化
            #     right_row_parts = []
            #     for num in added_value[buoy.id]:
            #         num_str = f"{int(num):3d}"  # 每个数字占3字符宽度
            #         right_row_parts.append(num_str)
            #     right_row = "".join(right_row_parts)
            #
            #     # 拼接并输出
            #     full_line = f"{left_row}  {right_row}"
            #     logger.info(full_line)
            # 拼接左矩阵（二进制字符串）
            # left_part = "".join(str(i) for i in buoy.success_rate_neighbor)
            # # 拼接右矩阵（格式化后的数字）
            # right_part = ""
            # for num in added_value[buoy.id]:
            #     num_formatted = f"{int(num):3d}"  # 数字宽度3，右对齐
            #     right_part += num_formatted
            # # 合并左、右矩阵，中间加间隔（如4个空格）
            # full_line = f"{left_part}    {right_part}"
            # # 用logger一次性输出
            # logger.info(full_line)
            # 第一步：拼接 success_rate_neighbor 的所有元素
            # success_str = "".join(str(i) for i in buoy.success_rate_neighbor)
            # logger.info(success_str)  # 去掉 end 参数，或直接拼接后续内容
            #
            # # 第二步：拼接间隔和 added_value 的内容
            # added_str = ""
            # # 先加两个空格间隔
            # added_str += "  "
            # # 再拼接格式化后的 added_value 整数
            # for val in added_value[buoy.id]:
            #     num = int(val)
            #     added_str += f"{num:3d} "  # 用字符串拼接代替 end=" "
            #
            # # 一次性输出所有内容
            # logger.info(success_str + added_str)



        # for buoy in self.buoys:
        #
        #     print(f"buoy.id:{buoy.id},position:{buoy.position}")








    def reset_lost_rate(self):
        for buoy in self.buoys:
            buoy.lost_rate = buoy.lost_rate_list[buoy.lost_idx]

            buoy.lost_idx += 1
            buoy.lost_rate = 0
            if buoy.lost_idx == 10:
                buoy.lost_idx = 0

    def transmit(self, sender, receiver):
        pass

    def move(self):
        for buoy in self.buoys:
            move_step_x = random.normalvariate(0, 0.001)
            move_step_y = random.normalvariate(0, 0.001)
            buoy.move(move_step_x, move_step_y)
    def update_topology(self,added_value):
        # 重新计算一跳和两跳邻居的邻接矩阵
        self.adj_matrix = self.build_adjacency_matrix()
        self.adj_matrix_two = self.build_adjacency_matrix_two()

        # 重新初始化邻居列表
        for buoy in self.buoys:
            buoy.neighbor = []
            buoy.neighbor_two = []
        self.initialize_neighbor()
        self.initialize_neighbor_two()

        # self.added_value =added_value
        self.build_s_rate(added_value)
