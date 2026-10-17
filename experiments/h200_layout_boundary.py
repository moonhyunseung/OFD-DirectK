"""Exact cache-layout diagnostic; group size, stored scales and decoded operator stay fixed.
This tests physical byte strides and optional XOR recoding, not a new quantizer.
Run: python experiments/h200_layout_boundary.py --out /absolute/new/output
"""
import argparse,hashlib,json,random,statistics,subprocess
from pathlib import Path
import torch,triton
import triton.language as tl
import h200_graph_boundary as base

EXPECTED_HELPER_SHA256="69573f2946d235656e78c6fdf4a0dfeadf2579eb110dd083b853b3d199c54047"
@triton.jit
def directk_groupmajor(Q,P,S,O,L:tl.constexpr,SIGNED_STORAGE:tl.constexpr,BT:tl.constexpr):
    t=tl.program_id(0)*BT+tl.arange(0,BT)
    h=tl.arange(0,16);j=tl.arange(0,32)
    acc=tl.full((16,BT),0,tl.float32)
    for g in tl.static_range(8):
        q=tl.load(Q+h[:,None]*256+g*32+j[None,:],h[:,None]<6,0)
        byte=tl.load(P+g*16*L+(j[:,None]//2)*L+t[None,:],t[None,:]<L,0).to(tl.int32)
        c=(byte>>((j[:,None]%2)*4))&15
        if SIGNED_STORAGE:n=c
        else:n=c^8
        code=tl.where(n>=8,n-16,n).to(tl.float16)
        dot=tl.dot(q,code)
        s=tl.load(S+g*L+t,t<L,0).to(tl.float32)
        acc=acc+dot*s[None,:]
    tl.store(O+h[:,None]*L+t[None,:],acc*0.0625,(h[:,None]<6)&(t[None,:]<L))

def run(q,pk,s,signed):
    L=pk.shape[-1];y=torch.empty(6,L,device='cuda')
    k=directk_groupmajor[(triton.cdiv(L,64),)](q,pk,s,y,L,signed,64,num_warps=4,enable_fp_fusion=False)
    return y,k

def measure(funcs,rng,ref=None):
    graphs={};outputs={};stream=torch.cuda.Stream()
    for name,fn in funcs.items():
        for _ in range(8):fn()
        torch.cuda.synchronize()
        g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):
            for _ in range(32):y=fn()
        g.replay();torch.cuda.synchronize()
        if ref is not None:torch.testing.assert_close(y,ref,atol=1e-4,rtol=2e-4)
        graphs[name]=g;outputs[name]=y
    samples={n:[] for n in graphs}
    for _ in range(31):
        names=list(graphs);rng.shuffle(names)
        for n in names:
            a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True)
            a.record();graphs[n].replay();b.record();b.synchronize();samples[n].append(a.elapsed_time(b)/32)
    return {n:dict(median_ms=statistics.median(ms),min_ms=min(ms),max_ms=max(ms),raw_ms=ms) for n,ms in samples.items()}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    out=a.out;out.mkdir(parents=True,exist_ok=False)
    helper=Path(base.__file__)
    assert hashlib.sha256(helper.read_bytes()).hexdigest()==EXPECTED_HELPER_SHA256
    torch.backends.cuda.matmul.allow_tf32=False;torch.set_float32_matmul_precision('highest');torch.set_num_threads(1)
    cfg=dict(status='RUNNING',scope='exact physical-layout experiment; resident repeated operand; not full serving',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=EXPECTED_HELPER_SHA256,
        source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=helper.parents[1],text=True).strip(),
        source_dirty=bool(subprocess.check_output(['git','status','--porcelain'],cwd=helper.parents[1],text=True).strip()),
        gpu=torch.cuda.get_device_name(),torch=torch.__version__,triton=triton.__version__,shape='B=1,KV=1,r=6,d=256,G=8,m=32',
        heldout_lengths=[2048,16384,65536,262144],heldout_seeds=[1741,1742,1743],numerical_seed=1740,
        tile=64,num_warps=4,graph_operations=32,samples=31,warmup=8,
        representation='offset [L,8,16] -> byte-stride permutation [8,16,L]; scales [L,8] -> [8,L]; signed optional XOR 0x88',
        numerical_budget='same diagnostic 64*eps32*absolute_term_envelope+1e-6; not a proven production bound',
        precision='FP16 Q/scales; FP32 accumulation/output, no TF32, no fp fusion',
        preprocessing='full-cache transpose/recode measured separately; append cost not measured; no amortized serving claim')
    def save():(out/'MANIFEST.json').write_text(json.dumps(cfg,indent=2))
    save();torch.manual_seed(1740);checks=[];kernels={}
    try:
        for L in [1,17,129,1024,8192]:
            for case in ['random','zero_code8','alternating','max_scale','tiny_scale']:
                c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32)
                if case=='zero_code8':c.fill_(8)
                if case=='alternating':c=(torch.arange(32,device='cuda')%2*15).expand(L,8,32).contiguous().int()
                s=(torch.rand(L,8,device='cuda')+.01).half()
                if case=='max_scale':s.fill_(65504)
                if case=='tiny_scale':s.fill_(2**-24)
                q=torch.randn(6,256,device='cuda').half();pk=base.pack(c)
                pg=pk.permute(1,2,0).contiguous();sg=s.T.contiguous();ps=pg^0x88
                assert torch.equal((ps^0x88).permute(2,0,1),pk)
                kd=((c.double()-8)*s.double()[:,:,None]).reshape(L,256)
                ref=q.double()@kd.T/16;env=q.double().abs()@kd.abs().T/16
                bound=64*torch.finfo(torch.float32).eps*env+1e-6
                for signed,data in [(False,pg),(True,ps)]:
                    y,k=run(q,data,sg,signed);kernels[str(signed)]=k
                    err=(y.double()-ref).abs();passed=bool(torch.isfinite(y).all() and (err<=bound).all())
                    checks.append(dict(length=L,case=case,signed_storage=signed,passed=passed,max_abs=err.max().item(),max_budget_ratio=(err/bound).max().item()))
        (out/'numerical.json').write_text(json.dumps(checks,indent=2))
        (out/'instruction_audit.json').write_text(json.dumps({n:base.audit(k) for n,k in kernels.items()},indent=2))
        for n,k in kernels.items():
            for kind in ['ptx','llir']:(out/(n+'.'+kind)).write_text(k.asm[kind])
        assert all(x['passed'] for x in checks),'Diagnostic gate failed'
        rows=[];rng=random.Random(1741)
        for seed in cfg['heldout_seeds']:
            torch.manual_seed(seed)
            for L in cfg['heldout_lengths']:
                c=torch.randint(0,16,(L,8,32),device='cuda',dtype=torch.int32);pk=base.pack(c)
                s=(torch.rand(L,8,device='cuda')+.01).half();q=torch.randn(6,256,device='cuda').half();q32=q.float()
                pg=pk.permute(1,2,0).contiguous();sg=s.T.contiguous();ps=pg^0x88;dense=base.decode(pk,s)
                funcs=dict(tensor_original=lambda:base.invoke('tensor_signed',q,None,pk,s)[0],
                    groupmajor_inline=lambda:run(q,pg,sg,False)[0],groupmajor_signed=lambda:run(q,ps,sg,True)[0],
                    decode_first=lambda:q32@base.decode(pk,s).T/16,predecoded_oracle=lambda:q32@dense.T/16)
                measured=measure(funcs,rng,q32@dense.T/16)
                pre=measure(dict(groupmajor_inline=lambda:(pk.permute(1,2,0).contiguous(),s.T.contiguous()),
                    groupmajor_signed=lambda:(pk.permute(1,2,0).contiguous()^0x88,s.T.contiguous())),rng)
                row=dict(seed=seed,length=L,timings=measured,preprocessing=pre)
                rows.append(row);(out/'timings.json').write_text(json.dumps(rows,indent=2))
                print(json.dumps(dict(seed=seed,L=L,ms={n:v['median_ms'] for n,v in measured.items()},pre_ms={n:v['median_ms'] for n,v in pre.items()})),flush=True)
        cfg.update(status='PASS',numerical_cases=len(checks),timing_rows=len(rows));save();print('LAYOUT_PASS',out,flush=True)
    except Exception as e:cfg.update(status='FAIL',error=repr(e));save();raise
if __name__=='__main__':main()
