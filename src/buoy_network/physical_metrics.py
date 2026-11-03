"""Read-only physical event metrics, separate from legacy penalized CSV fields."""
import math


def percentile95(values):
    return sorted(values)[math.ceil(.95*len(values))-1] if values else None


class PacketMetrics:
    def __init__(self, packets):
        self.packets = {}
        for p in packets:
            if p.id in self.packets or len(p.route) != 1 or p.if_sink:
                raise ValueError("Physical metrics require unique new, unresolved packets")
            self.packets[p.id] = {"route": list(p.route), "destination": p.destination,
                                  "retry": False, "pending": None, "last_step": -1, "delay": None}
        self.attempts = self.successes = self.failures = self.retries = self.loops = 0

    def attempt(self, packet_id, sender, receiver, success):
        p = self.packets[packet_id]
        if p["delay"] is not None or p["route"][-1] != sender or p["pending"] is not None:
            raise ValueError("Physical attempt packet/node mismatch")
        self.attempts += 1
        self.retries += int(p["retry"])
        p["retry"] = not success
        if success:
            self.successes += 1
            p["pending"] = receiver
        else:
            self.failures += 1

    def transferred(self, packet, step):
        p = self.packets[packet.id]
        if p["pending"] != packet.agent_id or p["last_step"] >= step or packet.route != p["route"]+[packet.agent_id]:
            raise ValueError("Physical transfer has no matching ACK or moved twice per slot")
        self.loops += int(packet.agent_id in p["route"])
        p["route"].append(packet.agent_id)
        p["pending"], p["last_step"] = None, step
        if packet.if_sink:
            if packet.agent_id != p["destination"]:
                raise ValueError("Delivery destination mismatch")
            p["delay"] = step

    def summary(self, episode, horizon):
        if any(p["pending"] is not None for p in self.packets.values()):
            raise ValueError("Unmatched successful physical transfer")
        delivered = [p for p in self.packets.values() if p["delay"] is not None]
        hops, delays = [len(p["route"])-1 for p in delivered], [p["delay"] for p in delivered]
        generated, n = len(self.packets), len(delivered)
        if any(delay > horizon for delay in delays):
            raise ValueError("Delivery after horizon")
        return {"schema": "original_tg_physical_v1", "episode": episode, "horizon": horizon,
                "generated_packets": generated, "delivered_packets": n, "unresolved_packets": generated-n,
                "delivery_rate": n/generated if generated else None,
                "unresolved_fraction": (generated-n)/generated if generated else None,
                "true_drop_count": None, "true_drop_rate": None,
                "success_hops_mean": sum(hops)/n if n else None,
                "success_delay_mean": sum(delays)/n if n else None, "success_delay_p95": percentile95(delays),
                "attempts": self.attempts, "ack_successes": self.successes, "ack_failures": self.failures,
                "retry_attempts": self.retries, "loop_transfers": self.loops,
                "success_hops": hops, "success_delays": delays}
