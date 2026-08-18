import argparse,csv,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src")); sys.path.insert(0,str(Path(__file__).resolve().parent))
from causal_opt.experiments.config import ExperimentConfig
from causal_opt.experiments.runner import run_experiment
from _dynamic_run import run_case

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--case",type=int,choices=[1,2,3,4],required=True); ap.add_argument("--config"); ap.add_argument("--seed",type=int,nargs="+"); ap.add_argument("--seed-start",type=int); ap.add_argument("--seed-end",type=int); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--resume",action="store_true"); ap.add_argument("--continue-on-error",action="store_true"); ap.add_argument("--output-dir",default="results"); a=ap.parse_args()
    config=a.config or f"configs/case{a.case}{'_smoke' if a.case>2 and a.smoke else ''}.yaml"
    cfg=ExperimentConfig.load(config)
    if a.seed_start is not None: seeds=range(a.seed_start,(a.seed_end if a.seed_end is not None else a.seed_start)+1)
    elif a.seed: seeds=range(a.seed[0],a.seed[1]+1) if len(a.seed)==2 else a.seed
    else: seeds=[cfg.seed]
    summary=Path(a.output_dir)/"dynamic_summary.csv"; summary.parent.mkdir(parents=True,exist_ok=True)
    for seed in seeds:
        started=time.perf_counter()
        try:
            if a.case in (3,4): folder,status=run_case(config,a.case,"all",a.output_dir,seed,a.smoke,a.resume); state="ok" if status["methods"].get("ours")=="ok" else "failed"
            else:
                cfg.seed=seed
                if a.smoke: cfg.n=min(cfg.n,200); cfg.d=min(cfg.d,5); cfg.s0=min(cfg.s0,5)
                folder,metrics=run_experiment(cfg,a.output_dir); state=metrics["status"]
            row={"case":a.case,"seed":seed,"status":state,"path":str(folder),"wall_seconds":time.perf_counter()-started}
        except Exception as exc:
            row={"case":a.case,"seed":seed,"status":"failed","path":"","wall_seconds":time.perf_counter()-started,"error":repr(exc)}
            if not a.continue_on_error: raise
        fields=["case","seed","status","path","wall_seconds","error"]
        exists=summary.exists()
        with summary.open("a",newline="") as f: w=csv.DictWriter(f,fieldnames=fields); (w.writeheader() if not exists else None); w.writerow({x:row.get(x,"") for x in fields})
        print(row)
if __name__=="__main__": main()
