import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src")); sys.path.insert(0,str(Path(__file__).resolve().parent))
from _dynamic_run import run_case
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/case4_smoke.yaml"); ap.add_argument("--seed",type=int); ap.add_argument("--smoke",action="store_true"); ap.add_argument("--method",choices=["ours","lpcmci","liegeois","all"],default="ours"); ap.add_argument("--output-dir",default="results"); ap.add_argument("--resume",action="store_true"); a=ap.parse_args()
    folder,status=run_case(a.config,4,a.method,a.output_dir,a.seed,a.smoke,a.resume); print(folder); print(status)
if __name__=="__main__": main()
