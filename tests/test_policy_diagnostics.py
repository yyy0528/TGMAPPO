import unittest
from types import SimpleNamespace
from src.buoy_network.policy_diagnostics import canonical_cycle,route_cycles,reachable,observed_closed_components,TraceAnalysis


class DiagnosticFunctions(unittest.TestCase):
    def test_cycles_preserve_direction_and_ignore_path_prefix(self):
        self.assertEqual(canonical_cycle([8,3]),(3,8))
        self.assertNotEqual(canonical_cycle([1,2,3]),canonical_cycle([1,3,2]))
        self.assertEqual(route_cycles([0,3,8,3,8,3])[(3,8)],2)
        self.assertFalse(route_cycles([0,1,2,3]))

    def test_observed_traps_are_not_physical_disconnections(self):
        edges = [(0,1),(1,2),(2,1),(3,4),(4,3),(4,24)]
        self.assertEqual(observed_closed_components(edges,24),[[1,2]])
        self.assertIn(24,reachable({1:[2,24],2:[1]},1))

    def test_active_wait_sink_opportunity_and_initial_agreement(self):
        analysis = TraceAnalysis(3,2)
        packets = [SimpleNamespace(agent_id=0,if_sink=False)]
        analysis.decision([1,3,3],[[0,.51,.49,0],[0,0,0,1],[0,0,0,1]],
            [[0,.51,.49,0],[0,0,0,1],[0,0,0,1]],[[0,1,1,1],[0,0,0,1],[0,0,0,1]],
            [[0,0],[1,0],[1,1]],packets,1.5)
        result = analysis.summary()
        self.assertEqual(result["active_decisions"],1)
        self.assertEqual(result["forced_wait_count"],2)
        self.assertEqual(result["active_wait_count"],0)
        self.assertEqual(result["sink_legal_count"],1)
        self.assertEqual(result["sink_chosen_count"],0)
        self.assertEqual(result["physical_sink_reachable_fraction"],1)
        self.assertEqual(result["initial_greedy_agreement"],1)

    def test_successful_loop_counts_and_wait_are_separate(self):
        analysis = TraceAnalysis(3,2)
        analysis.attempt(0,1,True,[[0,0],[1,0],[2,0]])
        analysis.attempt(1,0,True,[[0,0],[1,0],[2,0]])
        analysis.attempt(0,1,False,[[0,0],[1,0],[2,0]])
        analysis.finish_episode({"generated_packets":1,"delivered_packets":0,"loop_transfers":1},
            {7:{"route":[0,1,0],"delay":None}})
        result = analysis.summary()
        self.assertEqual(result["ack_successes"],2)
        self.assertEqual(result["ack_failures"],1)
        self.assertEqual(result["revisit_fraction_of_successes"],.5)
        self.assertEqual(result["top_closed_cycles"],[{"nodes":[0,1],"count":1}])
        self.assertEqual(result["episodes"][0]["unresolved_examples"][0]["route"],[0,1,0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
