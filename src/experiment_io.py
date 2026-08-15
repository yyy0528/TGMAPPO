"""Shared result paths and metric logging for all experiment families."""

import csv
import os


METRIC_FIELDS = (
    "episode",
    "delivery_rate",
    "average_hops",
    "average_delay",
)


def algorithm_slug(name):
    return name.strip().upper().replace("-", "_")


def moving_buoy_ids(total_buoys, num_moving_buoys):
    """Return exactly N evenly distributed buoy IDs."""
    if num_moving_buoys < 0 or num_moving_buoys > total_buoys:
        raise ValueError(
            "num_moving_buoys must be between 0 and {}".format(total_buoys)
        )
    if num_moving_buoys == 0:
        return set()
    return set(
        (index * total_buoys) // num_moving_buoys
        for index in range(num_moving_buoys)
    )


def metrics_path(output_root, num_moving_buoys, algorithm_name):
    log_dir = os.path.join(
        output_root,
        "moves_{}".format(num_moving_buoys),
        "logs",
    )
    os.makedirs(log_dir, exist_ok=True)
    return os.path.join(
        log_dir,
        "{}_metrics.csv".format(algorithm_slug(algorithm_name)),
    )


class MetricCsvWriter:
    """Write one comparable metric row per completed episode."""

    def __init__(self, output_root, num_moving_buoys, algorithm_name):
        self.path = metrics_path(
            output_root, num_moving_buoys, algorithm_name
        )
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=METRIC_FIELDS)
        self._writer.writeheader()
        self._file.flush()

    def write(self, episode, delivery_rate, average_hops, average_delay):
        self._writer.writerow(
            {
                "episode": episode,
                "delivery_rate": delivery_rate,
                "average_hops": average_hops,
                "average_delay": average_delay,
            }
        )
        self._file.flush()

    def close(self):
        if not self._file.closed:
            self._file.close()
