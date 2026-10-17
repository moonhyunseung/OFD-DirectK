"""Signed DirectK floating-point boundary characterization.
No absolute 1e-6 floor: error is normalized by absolute decoded contraction terms.
The diagnostic gamma64 budget assumes FP32 arithmetic without unexpected input flushing;
a failure is retained and blocks any assertion that all FP16 inputs obey that budget.
"""
import argparse,hashlib,json,subprocess
from pathlib import Path
import torch
import h200_graph_boundary as base
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    expected='69573f2946d235656e78c6fdf4a0dfeadf2579eb110dd083b853b3d199c54047'
    assert hashlib.sha256(Path(base.__file__).read_bytes()).hexdigest()==expected
    torch.manual_seed(1750);torch.backends.cuda.matmul.allow_tf32=False
    u32=torch.finfo(torch.float32).eps/2;u64=torch.finfo(torch.float64).eps/2
    gamma=64*u32/(1-64*u32)+256*u64/(1-256*u64)
    rows=[]
    for L in [1,17,129,1024]:
        for case in ['normal','min_q','min_scale','max_q','max_scale','both_max','mixed_exponents','alternating_cancel']:
            for codes_case in ['random','all_minus8','all_plus7','zero_code8']:
                c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32)
                if codes_case=='all_minus8':c.fill_(0)
                if codes_case=='all_plus7':c.fill_(15)
                if codes_case=='zero_code8':c.fill_(8)
                q=torch.randn(6,256,device='cuda').half();s=(torch.rand(L,8,device='cuda')+.01).half()
                if case=='min_q':q.fill_(2**-24)
                if case=='min_scale':s.fill_(2**-24)
                if case in ['max_q','both_max']:q.fill_(65504)
                if case in ['max_scale','both_max']:s.fill_(65504)
                if case=='mixed_exponents':
                    q=(torch.randint(0,2,(6,256),device='cuda')*2-1).half()*torch.pow(2.,torch.randint(-24,16,(6,256),device='cuda')).half()
                    s=torch.pow(2.,torch.randint(-24,16,(L,8),device='cuda')).half()
                if case=='alternating_cancel':q=(torch.arange(256,device='cuda')%2*2-1).half().expand(6,256).contiguous()
                pk=base.pack(c);kd=((c.double()-8)*s.double()[:,:,None]).reshape(L,256)
                ref=q.double()@kd.T/16;env=q.double().abs()@kd.abs().T/16
                bound=gamma*env
                for method in ['signed','tensor_signed']:
                    y,_=base.invoke(method,q,None,pk,s);err=(y.double()-ref).abs()
                    ratio=torch.where(bound>0,err/bound,torch.where(err==0,0.,float('inf')))
                    norm=torch.where(env>0,err/env,torch.where(err==0,0.,float('inf')))
                    rows.append(dict(length=L,case=case,codes_case=codes_case,method=method,passed=bool(torch.isfinite(y).all() and (err<=bound).all()),max_abs=err.max().item(),max_normalized_error=norm.max().item(),max_budget_ratio=ratio.max().item()))
    failures=[x for x in rows if not x['passed']]
    m=dict(status='PASS' if not failures else 'BOUNDARY_FOUND',cases=len(rows),failure_count=len(failures),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=expected,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),
        gamma64=gamma,absolute_floor=0.,seed=1750,gpu=torch.cuda.get_device_name(),torch=torch.__version__,
        interpretation='Conditional floating-point diagnostic, not a universal hardware proof or downstream quality guarantee',
        failures=failures)
    (out/'rows.json').write_text(json.dumps(rows,indent=2));(out/'MANIFEST.json').write_text(json.dumps(m,indent=2))
    print(json.dumps(m,indent=2),flush=True)
if __name__=='__main__':main()
