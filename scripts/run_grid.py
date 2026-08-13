import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from causal_opt.experiments.config import ExperimentConfig
from causal_opt.experiments.runner import run_experiment

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--case",type=int,required=True); ap.add_argument("--config")
    ap.add_argument("--seed",type=int,nargs="+"); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--resume",action="store_true")
    a=ap.parse_args(); cfg=ExperimentConfig.load(a.config or f"configs/case{a.case}.yaml")
    seeds=a.seed or [cfg.seed]
    for seed in range(seeds[0],seeds[1]+1) if len(seeds)==2 else seeds:
        cfg.seed=seed
        if a.smoke: cfg.n=min(cfg.n,200); cfg.d=min(cfg.d,5); cfg.s0=min(cfg.s0,5)
        folder,metrics=run_experiment(cfg); print(folder,metrics["status"])
if __name__=="__main__": main()
