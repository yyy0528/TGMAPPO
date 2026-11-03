"""Pure trace analysis. No action/reward replacement or simulator mutation."""
import math
from collections import Counter


def canonical_cycle(cycle):
    values = tuple(cycle)
    if not values:
        raise ValueError("Empty cycle")
    return min(values[index:]+values[:index] for index in range(len(values)))


def route_cycles(route):
    """Count simple closed walks in the loop-erased running path; WAIT is not a hop."""
    stack, counts = [], Counter()
    for node in route:
        if node in stack:
            index = stack.index(node)
            counts[canonical_cycle(stack[index:])] += 1
            stack = stack[:index+1]
        else:
            stack.append(node)
    return counts


def reachable(graph, source):
    seen, pending = set(), [source]
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(graph.get(node, []))
    return seen


def observed_closed_components(edges, sink):
    """SCCs closed in observed successful-edge graph, not all physically possible edges."""
    graph = {}
    for source, target in edges:
        graph.setdefault(source, set()).add(target)
        graph.setdefault(target, set())
    closure = {node:reachable(graph, node) for node in graph}
    remaining, components = set(graph), []
    while remaining:
        first = min(remaining)
        group = {node for node in remaining if node in closure[first] and first in closure[node]}
        remaining.difference_update(group)
        if sink not in group and (len(group) > 1 or first in graph[first]):
            outgoing = sorted({target for node in group for target in graph[node] if target not in group})
            if not outgoing:
                components.append(sorted(group))
    return sorted(components)


class TraceAnalysis:
    def __init__(self, n=25, sink=24):
        self.n, self.sink = n, sink
        self.active = self.active_wait = self.forced_wait = 0
        self.sink_opportunities = self.sink_chosen = 0
        self.initial_agreement = self.choices_checked = 0
        self.entropy = self.gaps = self.sink_probability = 0.
        self.physical_reachable = self.live_packet_slots = 0
        self.attempts = self.ack_successes = self.ack_failures = self.away_moves = 0
        self.actions = [[0]*(n+1) for _ in range(n)]
        self.edges, self.episodes = [], []
        self.closed_cycles = Counter()

    def decision(self, actions, probabilities, initial_probabilities, mask, positions, packets, radius):
        graph = {node:[other for other in range(self.n) if node != other and math.hypot(
            positions[node][0]-positions[other][0], positions[node][1]-positions[other][1]) <= radius]
                 for node in range(self.n)}
        can_reach = {node:self.sink in reachable(graph, node) for node in range(self.n)}
        for p in packets:
            if not p.if_sink:
                self.live_packet_slots += 1
                self.physical_reachable += int(can_reach[p.agent_id])
        for node, action in enumerate(actions):
            legal = [i for i,value in enumerate(mask[node]) if value]
            if len(legal) == 1:
                self.forced_wait += int(action == self.n)
                continue
            self.active += 1
            self.active_wait += int(action == self.n)
            self.actions[node][action] += 1
            ranked = sorted((probabilities[node][i] for i in legal), reverse=True)
            self.gaps += ranked[0]-ranked[1]
            self.entropy += -sum(p*math.log(max(p,1e-30)) for p in ranked)/math.log(len(legal))
            if initial_probabilities is not None:
                a = max(legal, key=lambda i:probabilities[node][i])
                b = max(legal, key=lambda i:initial_probabilities[node][i])
                self.initial_agreement += int(a == b)
                self.choices_checked += 1
            if self.sink in legal:
                self.sink_opportunities += 1
                self.sink_chosen += int(action == self.sink)
                self.sink_probability += probabilities[node][self.sink]

    def attempt(self, sender, receiver, success, positions):
        self.attempts += 1
        self.ack_successes += int(success)
        self.ack_failures += int(not success)
        if success:
            self.edges.append((sender,receiver))
            sink = positions[self.sink]
            before = math.hypot(positions[sender][0]-sink[0],positions[sender][1]-sink[1])
            after = math.hypot(positions[receiver][0]-sink[0],positions[receiver][1]-sink[1])
            self.away_moves += int(after >= before-1e-9)

    def finish_episode(self, physical, packets):
        examples = []
        for key, packet in sorted(packets.items()):
            cycles = route_cycles(packet["route"])
            self.closed_cycles.update(cycles)
            if packet["delay"] is None:
                examples.append({"packet_id":key,"source":packet["route"][0],"final_node":packet["route"][-1],
                    "route":packet["route"],"hops":len(packet["route"])-1,
                    "closed_cycles":[{"nodes":list(c),"count":count} for c,count in cycles.most_common(3)]})
        self.episodes.append({"physical":physical,"unresolved_examples":sorted(examples,key=lambda row:-row["hops"])[:3]})

    def summary(self):
        delivered = sum(row["physical"]["delivered_packets"] for row in self.episodes)
        generated = sum(row["physical"]["generated_packets"] for row in self.episodes)
        loops = sum(row["physical"]["loop_transfers"] for row in self.episodes)
        return {"generated":generated,"delivered":delivered,"active_decisions":self.active,
                "active_wait_count":self.active_wait,"active_wait_fraction":self.active_wait/max(self.active,1),
                "forced_wait_count":self.forced_wait,"sink_legal_count":self.sink_opportunities,"sink_chosen_count":self.sink_chosen,
                "mean_sink_probability_when_legal":self.sink_probability/max(self.sink_opportunities,1),
                "initial_greedy_agreement":self.initial_agreement/max(self.choices_checked,1),
                "initial_comparison_count":self.choices_checked,"mean_normalized_entropy":self.entropy/max(self.active,1),
                "mean_top_probability_gap":self.gaps/max(self.active,1),"attempts":self.attempts,
                "ack_successes":self.ack_successes,"ack_failures":self.ack_failures,
                "revisit_transfers":loops,"revisit_fraction_of_successes":loops/max(self.ack_successes,1),
                "nonprogress_successes":self.away_moves,
                "physical_sink_reachable_fraction":self.physical_reachable/max(self.live_packet_slots,1),
                "live_packet_slots":self.live_packet_slots,"observed_closed_sccs":observed_closed_components(self.edges,self.sink),
                "top_closed_cycles":[{"nodes":list(c),"count":count} for c,count in self.closed_cycles.most_common(8)],
                "action_counts_by_sender":self.actions,"episodes":self.episodes}
