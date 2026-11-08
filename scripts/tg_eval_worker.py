"""Native GPU evaluation with deterministic settings matching the parent study."""
import os,sys
from pathlib import Path
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
import run_algorithm as launcher
from buoy_network.main import main
known,extra=launcher.build_parser().parse_known_args()
if known.algorithm!='tg-mappo':raise ValueError('Original TG only')
output=launcher.create_run_root(known.output_root,known.run_name)
main(launcher.with_tg_mappo_defaults(extra+['--output_root',str(output)]))
