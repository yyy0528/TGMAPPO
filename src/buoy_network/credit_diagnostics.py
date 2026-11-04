"""Pure, read-only summaries of sampled team credit; no policy correction."""
import numpy as np


def advantages(returns, values, active_masks):
    returns, values, active = np.asarray(returns), np.asarray(values), np.asarray(active_masks) > 0
    if returns.shape != values.shape or active.shape != returns.shape:
        raise ValueError("Credit arrays must have identical shapes")
    raw = returns-values
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite credit values")
    if active.any():
        normalized = (raw-raw[active].mean())/(raw[active].std()+1e-5)
        normalized[~active] = 0
    else:
        normalized = np.zeros_like(raw)
    return raw, normalized


def summarize(rows):
    groups = {
        "all_active": lambda r: True,
        "sink_chosen": lambda r: r["action"] == r["sink"],
        "sink_legal_not_chosen": lambda r: r["sink_legal"] and r["action"] != r["sink"],
        "wait": lambda r: r["action"] == r["wait"],
        "successful_revisit": lambda r: r["ack"] is True and r["revisit"],
        "successful_new_node": lambda r: r["ack"] is True and not r["revisit"],
        "revisit_with_other_delivery": lambda r: r["ack"] is True and r["revisit"] and r["other_deliveries"] > 0,
        "failed_attempt": lambda r: r["ack"] is False,
    }
    result = {}
    for name,predicate in groups.items():
        selected = [r for r in rows if predicate(r)]
        result[name] = {"count":len(selected)}
        if selected:
            for field in ("advantage_raw","advantage_normalized","return","value","reward",
                          "p_selected_change","p_sink_change"):
                result[name][field+"_mean"] = float(np.mean([r[field] for r in selected]))
            result[name]["positive_advantage_count"] = sum(r["advantage_normalized"] > 0 for r in selected)
            result[name]["selected_probability_increased_count"] = sum(r["p_selected_change"] > 0 for r in selected)
    return result
