import math
import random
import numpy as np

NUM = 100

class buoy_node:
    def __init__(self):
        self.id = None
        self.energy = None
        self.o_energy = None
        self.position = None
        self.neighbor = []
        self.neighbor_two = []
        self.packet = 0
        self.if_sink = False
        self.packet_buffer = None
        self.lost_rate_list = []
        self.lost_rate = 0
        self.success_rate_neighbor = []
        self.success_number = []
        self.send_number = []
        self.lost_idx = 0
        self.confirmations = []
        self.hop_count = 1

    def initialize(self, buoy_id, energy, position):
        self.id = buoy_id
        self.energy = energy
        self.o_energy = energy
        self.position = position
        for i in range(10):
            random_int = random.randint(1, 10)
            self.lost_rate_list.append(random_int * (self.hop_count / 20))
        self.lost_rate = self.lost_rate_list[0]
        self.lost_idx += 1

    def get_info(self):
        info = {'id': self.id, 'energy': self.energy, 'position': self.position}
        return info

    def note_transmit(self):
        pass

    def send(self):
        if self.packet > 0:
            self.packet -= 1
            return True
        return False

    def receive(self):
        self.hop_count += 1
        if self.hop_count < len(self.lost_rate_list):
            self.lost_rate = self.lost_rate_list[self.hop_count]
        else:
            self.lost_rate = self.lost_rate_list[-1]

        rate = random.random()
        if rate >= self.lost_rate:
            self.packet += 1
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
        self.packet = 0
        self.lost_rate = self.lost_rate_list[0]
        self.lost_idx = 1
        self.hop_count = 1

    def move(self, dx, dy, dz):
        self.position[0] += dx
        self.position[1] += dy
        self.position[2] += dz
        return

class data_packet:
    def __init__(self, packet_id, source, destination):
        self.id = packet_id
        self.agent_id = source
        self.last_pos = source
        self.stay_time = 1
        self.memories = [0]
        self.route = []
        self.life = 80
        self.route.append(source)
        self.destination = destination
        self.if_sink = False
        self.change_n = 0
        self.q_time = [0]

    def transmit(self, next_hop):
        self.route.append(next_hop)
        self.q_time.append(self.stay_time)
        self.agent_id = next_hop
        if next_hop == self.destination:
            self.if_sink = True
            self.q_time.append(0)
        else:
            self.if_sink = False

    def add_memory(self, memory_id):
        self.memories.append(memory_id)

class buoy_net:
    def __init__(self, buoys):
        self.buoys = []
        self.count_NB = []
        self.buoys_number = len(buoys)
        random.seed(2568)
        self.initialize_node(buoys)

        self.adj_matrix = self.build_adjacency_matrix()
        self.adj_matrix_two = self.build_adjacency_matrix_two()
        self.initialize_neighbor()
        self.initialize_neighbor_two()
        self.build_s_rate()

    def initialize_node(self, buoys):
        for i in range(self.buoys_number):
            buoy = buoy_node()
            buoy.initialize(i, buoys[i][0], buoys[i][1:])
            self.buoys.append(buoy)

    def build_adjacency_matrix(self):
        neighbors = [[] for _ in range(self.buoys_number)]
        for i in range(self.buoys_number):
            for j in range(self.buoys_number):
                if i != j:  # Avoid self-loop
                    dist = math.sqrt(
                        math.pow((self.buoys[i].get_position()[0][0] - self.buoys[j].get_position()[0][0]),2) +
                        math.pow((self.buoys[i].get_position()[0][1] - self.buoys[j].get_position()[0][1]),2) +
                        math.pow((self.buoys[i].get_position()[0][2] - self.buoys[j].get_position()[0][2]),2)
                    )
                    if dist <= 1.5:
                        neighbors[i].append(j)
        for i, neighbor_list in enumerate(neighbors):
            self.count_NB.append(len(neighbor_list))
        return neighbors

    def build_adjacency_matrix_two(self):
        neighbors = [[] for _ in range(self.buoys_number)]
        for i in range(self.buoys_number):
            for j in range(self.buoys_number):
                if i != j:  # Avoid self-loop
                    dist = math.sqrt(
                        math.pow((self.buoys[i].get_position()[0][0] - self.buoys[j].get_position()[0][0]), 2) +
                        math.pow((self.buoys[i].get_position()[0][1] - self.buoys[j].get_position()[0][1]), 2) +
                        math.pow((self.buoys[i].get_position()[0][2] - self.buoys[j].get_position()[0][2]), 2)
                    )
                    if dist <= 3:
                        neighbors[i].append(j)
        return neighbors

    def initialize_neighbor(self):
        for buoy in self.buoys:
            buoy.neighbor = self.adj_matrix[buoy.id]  # Get neighbors from the adjacency matrix
            buoy.success_number = [0] * (len(buoy.neighbor) * 200)
            buoy.send_number = [0] * (len(buoy.neighbor) * 200)
            buoy.confirmations = [0] * (len(buoy.neighbor) * 200)

    def initialize_neighbor_two(self):
        for buoy in self.buoys:
            buoy.neighbor_two = self.adj_matrix_two[buoy.id]  # Get neighbors from the second adjacency matrix

    def get_buoys(self):
        return self.buoys

    def build_s_rate(self):
        for buoy in self.buoys:
            for neighbor_id in buoy.neighbor:
                success = sum(1 for _ in range(10) if random.random() >= self.buoys[neighbor_id].lost_rate)
                rate = success / 10
                buoy.success_rate_neighbor.append(rate)

    def reset_lost_rate(self):
        for buoy in self.buoys:
            buoy.lost_rate = buoy.lost_rate_list[buoy.lost_idx]
            buoy.lost_idx += 1
            if buoy.lost_idx == 10:
                buoy.lost_idx = 0

    def transmit(self, sender, receiver):
        success = self.buoys[sender].send()
        if success:
            self.buoys[receiver].receive()
            self.buoys[sender].send_number[sender] += 1
            self.buoys[receiver].success_number[receiver] += 1
        return success

    def move(self):
        for buoy in self.buoys:
            move_step_x = random.normalvariate(0, 0.001)
            move_step_y = random.normalvariate(0, 0.001)
            move_step_z = random.normalvariate(0, 0.001)
            buoy.move(move_step_x, move_step_y, move_step_z)

    def load_balanced_transmit(self, packet):
        current_buoy = packet.agent_id
        candidates = self.buoys[current_buoy].get_neighbor()

        best_next_hop = None
        highest_success_rate = -1
        for candidate in candidates:
            if self.buoys[candidate].success_rate_neighbor[candidate] > highest_success_rate:
                highest_success_rate = self.buoys[candidate].success_rate_neighbor[candidate]
                best_next_hop = candidate

        if best_next_hop is not None:
            packet.transmit(best_next_hop)
            self.transmit(current_buoy, best_next_hop)
