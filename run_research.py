"""Research entry point. The legacy run_algorithm.py entry remains independent."""
#研究 Graph-MAPPO 路径  为验证新模块设计的独立参考框架和测试环境
#
import sys

if sys.version_info < (3, 10):
    raise SystemExit("Research architecture requires Python >= 3.10; legacy entry points are unchanged.")

import sys

if sys.version_info < (3, 10):
    raise SystemExit("Research architecture requires Python >= 3.10; legacy entry points are unchanged.")

from src.buoy_network.research.cli import main


if __name__ == "__main__":
    raise SystemExit(main())
