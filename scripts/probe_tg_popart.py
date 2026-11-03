"""Small, isolated optimizer/normalization diagnostic; no environment or policy training."""
import argparse
import json
import hashlib
import importlib.util
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root",default="results/tg_popart_probe")
    parser.add_argument("--source-file",default=str(ROOT/"src/onpolicy/algorithms/utils/popart.py"))
    args = parser.parse_args()
    source = Path(args.source_file).resolve()
    spec = importlib.util.spec_from_file_location("isolated_popart",str(source))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    PopArt = module.PopArt
    torch.manual_seed(11)
    model = PopArt(3,1)
    optimizer = torch.optim.Adam(model.parameters(),lr=.01)
    registered = {id(p) for g in optimizer.param_groups for p in g["params"]}
    x = torch.tensor([[1.,2.,3.],[-1.,1.,0.]])
    with torch.no_grad():
        raw_before = model.denormalize(model(x)).copy()
    original_ids = {k:id(p) for k,p in model.named_parameters()}
    model.update(torch.tensor([[40.],[50.],[60.]]))
    with torch.no_grad():
        raw_after = model.denormalize(model(x)).copy()
    head = [model.weight.detach().clone(), model.bias.detach().clone()]
    loss = (model(x)-model.normalize(torch.tensor([[45.],[55.]]))).square().mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    report = {"torch":torch.__version__,"scope":"isolated PopArt, no Actor/environment/PPO",
              "source_file":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
              "head_parameters_in_optimizer":all(id(p) in registered for p in (model.weight,model.bias)),
              "parameter_identity_preserved":all(original_ids[k] == id(p) for k,p in model.named_parameters()),
              "denormalized_output_max_change":float(abs(raw_after-raw_before).max()),
              "head_has_nonzero_gradient":any(p.grad is not None and bool(p.grad.abs().max() > 0) for p in (model.weight,model.bias)),
              "head_changed_by_optimizer":any(not torch.equal(a,b) for a,b in zip(head,(model.weight,model.bias))),
              "normalization_parameters_require_grad":{k:p.requires_grad for k,p in model.named_parameters() if k not in ("weight","bias")}}
    root = Path(args.output_root).resolve()/datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    root.mkdir(parents=True,exist_ok=False)
    (root/"report.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(json.dumps(report,indent=2))
    print("Report: "+str(root/"report.json"))


if __name__ == "__main__":
    main()
