#!/usr/bin/env python3
import argparse, csv, json, logging, math, random, sys, time
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pdebench_reaction_diffusion_conditional_dsbm as base


def safe_torch_load(path, map_location="cpu"):
    """Load checkpoints safely across PyTorch versions."""
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def setup_logging(path):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.FileHandler(path,mode="a"),logging.StreamHandler(sys.stdout)],force=True)

def log(*args):
    logging.info(" ".join(str(x) for x in args))

def restore_rng_state(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"]); torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])

def load_response(path):
    obj=safe_torch_load(path,map_location="cpu")
    if not bool(obj.get("response_only",False)):
        raise RuntimeError(f"{path}: expected response_only=True")
    if bool(obj.get("endpoint_labels_included",True)):
        raise RuntimeError(f"{path}: endpoint labels unexpectedly included")
    return {
        "x0":obj["x0_norm"].float(),
        "a":obj["a"].float(),
        "direction":obj["direction"].float(),
        "Jv_star":obj["Jv_star_norm"].float(),
    }

class TangentFieldDSBM(base.ConditionalFieldDSBM):
    def __init__(self,cfg,state_shape,intervention_dim,device,lambda_sens,sens_every,sens_batch_size,anchor_fraction,sens_seed):
        super().__init__(cfg,state_shape,intervention_dim,device)
        self.lambda_sens=float(lambda_sens); self.sens_every=int(sens_every)
        self.sens_batch_size=int(sens_batch_size); self.anchor_fraction=float(anchor_fraction)
        self.sens_generator=torch.Generator(device="cpu").manual_seed(int(sens_seed))

    def _idx(self,n,b):
        return torch.randint(0,n,(b,),generator=self.sens_generator,device="cpu")

    def _sens_noise_bank(self,b,dtype):
        c,h,w=self.state_shape
        return [
            torch.randn(b,c,h,w,generator=self.sens_generator,dtype=dtype,device="cpu").to(self.device)
            for _ in range(self.cfg.num_steps)
        ]

    def _sample(self,src,b):
        if b<=0: return None
        i=self._idx(src["x0"].shape[0],b)
        return {k:src[k][i].to(self.device) for k in ["x0","a","direction","Jv_star"]}

    def tangent_training_rollout(self,x0,a,direction,noise_bank):
        dt=1.0/float(self.cfg.num_steps)
        x=x0; R=torch.zeros_like(x)
        for k in range(self.cfg.num_steps):
            t=torch.full((x.shape[0],1),k/float(self.cfg.num_steps),device=x.device,dtype=x.dtype)
            def drift_fn(xi,ai):
                return self.net_f(xi,ai,t)
            drift,td=torch.autograd.functional.jvp(
                drift_fn,(x,a),(R,direction),create_graph=True,strict=False
            )
            x=x+dt*drift+self.cfg.reference_sigma*math.sqrt(dt)*noise_bank[k]
            R=R+dt*td
        return R

    def pathwise_response_loss(self,anchor,colloc):
        total=self.sens_batch_size
        na=int(round(total*self.anchor_fraction))
        if total>=2: na=max(1,min(total-1,na))
        nc=total-na
        pieces=[p for p in [self._sample(anchor,na),self._sample(colloc,nc)] if p is not None]
        x0=torch.cat([p["x0"] for p in pieces],0)
        a=torch.cat([p["a"] for p in pieces],0)
        d=torch.cat([p["direction"] for p in pieces],0)
        target=torch.cat([p["Jv_star"] for p in pieces],0)
        perm=torch.randperm(x0.shape[0],generator=self.sens_generator,device="cpu").to(self.device)
        x0=x0[perm]; a=a[perm]; d=d[perm]; target=target[perm]
        pred=self.tangent_training_rollout(x0,a,d,self._sens_noise_bank(x0.shape[0],x0.dtype))
        return F.mse_loss(pred,target)

    def train_pass_tangent(self,data,fb,anchor,colloc):
        z0,z1,a=self.regenerate_coupling(data)
        net=self.nets[fb]; net.train()
        opt=torch.optim.AdamW(net.parameters(),lr=self.cfg.lr,weight_decay=1e-5)
        n=z0.shape[0]; rb=[]; rs=[]
        for step in range(1,self.cfg.inner_steps+1):
            b=min(self.cfg.batch_size,n)
            idx=torch.randint(0,n,(b,),device=self.device)
            zt,ba,t,target=self.get_train_tuple(z0[idx],z1[idx],a[idx],fb)
            pred=net(zt,ba,t); bridge=F.mse_loss(pred,target)
            sens=None
            if fb=="f" and self.lambda_sens>0 and step%self.sens_every==0:
                sens=self.pathwise_response_loss(anchor,colloc)
                total=bridge+self.lambda_sens*sens
            else:
                total=bridge
            opt.zero_grad(set_to_none=True); total.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(),self.cfg.grad_clip); opt.step()
            rb.append(float(bridge.detach().cpu())); rb=rb[-100:]
            if sens is not None:
                rs.append(float(sens.detach().cpu())); rs=rs[-100:]
            rep=max(50,self.cfg.inner_steps//5)
            if step==1 or step%rep==0 or step==self.cfg.inner_steps:
                log(f"{fb} step {step:5d}/{self.cfg.inner_steps}",
                    f"bridge={float(bridge.detach().cpu()):.6f}",
                    f"sens={(np.mean(rs) if rs else float('nan')):.6f}",
                    f"total={float(total.detach().cpu()):.6f}")
        self.prev_fb=fb
        return {"bridge_loss_last100":float(np.mean(rb)),
                "sensitivity_loss_last100":float(np.mean(rs)) if rs else None}

def gradient_sanity(model,anchor,colloc):
    old=model.sens_batch_size; st=model.sens_generator.get_state(); model.sens_batch_size=min(2,old)
    model.net_f.zero_grad(set_to_none=True); model.net_b.zero_grad(set_to_none=True)
    try:
        loss=model.pathwise_response_loss(anchor,colloc); loss.backward()
        fsq=0.0; bsq=0.0
        for p in model.net_f.parameters():
            if p.grad is not None: fsq+=float((p.grad.detach()**2).sum().cpu())
        for p in model.net_b.parameters():
            if p.grad is not None: bsq+=float((p.grad.detach()**2).sum().cpu())
        if fsq<=0: raise RuntimeError("zero forward sensitivity gradient")
        if bsq>0: raise RuntimeError("sensitivity unexpectedly touched backward net")
        return {"loss":float(loss.detach().cpu()),"forward_grad_norm":fsq**0.5}
    finally:
        model.net_f.zero_grad(set_to_none=True); model.net_b.zero_grad(set_to_none=True)
        model.sens_batch_size=old; model.sens_generator.set_state(st)


def subset_response(source, fraction, seed):
    """Random deterministic subset; preserves original response-domain support."""
    fraction = float(fraction)
    if not (0.0 < fraction <= 1.0):
        raise ValueError("--response-fraction must be in (0,1].")
    if fraction >= 1.0:
        return {k: v.clone() for k, v in source.items()}

    n = source["x0"].shape[0]
    keep = max(1, int(round(n * fraction)))
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    idx = torch.randperm(n, generator=g)[:keep]
    return {k: v[idx].clone() for k, v in source.items()}


def corrupt_response_targets(source, mode, seed):
    """
    Corrupt only Jv_star while preserving the operating point
    (x0, a, intervention direction). Applied independently to anchor
    and continuous response sets.
    """
    out = {k: v.clone() for k, v in source.items()}

    if mode == "correct":
        return out

    if mode == "signflip":
        out["Jv_star"] = -out["Jv_star"]
        return out

    if mode == "shuffle":
        n = out["Jv_star"].shape[0]
        g = torch.Generator(device="cpu").manual_seed(int(seed))
        perm = torch.randperm(n, generator=g)
        # Avoid the rare identity permutation for tiny sets.
        if n > 1 and torch.equal(perm, torch.arange(n)):
            perm = torch.roll(perm, shifts=1)
        out["Jv_star"] = out["Jv_star"][perm].clone()
        return out

    raise ValueError(mode)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data-dir",default="runs/pdebench_reaction_diffusion_official")
    p.add_argument("--baseline-run-root",default="runs/pdebench_rd_conditional_sigma_0p15_imf1")
    p.add_argument("--run-root",required=True)
    p.add_argument("--seed",type=int,default=32)
    p.add_argument("--fork-imf",type=int,default=1)
    p.add_argument("--total-imf",type=int,default=2)
    p.add_argument("--lambda-sens",type=float,required=True)
    p.add_argument("--sens-every",type=int,default=10)
    p.add_argument("--sens-batch-size",type=int,default=4)
    p.add_argument("--anchor-fraction",type=float,default=0.25)
    p.add_argument("--response-fraction",type=float,default=1.0,
                   help="Fraction of CONTINUOUS response_collocation points to use. Anchor response set is unchanged.")
    p.add_argument("--target-mode",choices=["correct","shuffle","signflip"],default="correct",
                   help="Corrupt response targets for target-correctness ablation.")
    p.add_argument("--device",default=None)
    args=p.parse_args()

    device=torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    data_dir=Path(args.data_dir)
    endpoint,response_eval,metadata,state_shape,intervention_dim=base.load_dataset(data_dir)
    fork_path=Path(args.baseline_run_root)/f"seed_{args.seed}"/f"imf_{args.fork_imf}.pt"
    ckpt=safe_torch_load(fork_path,map_location="cpu")
    cfg=base.Config(**ckpt["config"]); cfg.fork_imf=args.fork_imf; cfg.total_imf=args.total_imf
    cfg.finite_delta=float(metadata.get("finite_delta",cfg.finite_delta))
    anchor=load_response(data_dir/"anchor_response.pt")
    colloc=load_response(data_dir/"response_collocation.pt")

    # Response-amount ablation changes ONLY continuous collocation count.
    colloc=subset_response(
        colloc,
        args.response_fraction,
        seed=args.seed + 810001,
    )

    # Target-correctness ablation corrupts the target alignment in BOTH
    # response-supervision sources, while leaving endpoint data unchanged.
    anchor=corrupt_response_targets(
        anchor,
        args.target_mode,
        seed=args.seed + 820001,
    )
    colloc=corrupt_response_targets(
        colloc,
        args.target_mode,
        seed=args.seed + 830001,
    )

    model=TangentFieldDSBM(cfg,state_shape,intervention_dim,device,args.lambda_sens,args.sens_every,args.sens_batch_size,args.anchor_fraction,args.seed+700001)
    model.load_state_dict(ckpt["model"]); restore_rng_state(ckpt["rng_state"])

    run_dir=Path(args.run_root)/f"seed_{args.seed}"; run_dir.mkdir(parents=True,exist_ok=True)
    setup_logging(run_dir/"tangent_sbm.log")
    log("="*80); log("PDEBENCH REACTION-DIFFUSION TANGENT-SBM")
    log(f"seed={args.seed}",f"fork_imf={args.fork_imf}",f"total_imf={args.total_imf}",f"lambda={args.lambda_sens}")
    log(f"response_fraction={args.response_fraction}",f"target_mode={args.target_mode}",
        f"anchor_n={anchor['x0'].shape[0]}",f"collocation_n={colloc['x0'].shape[0]}")

    shared=base.convergence(model,endpoint["train"],cfg,device)
    log("Shared-fork train endpoint RMSE(norm)=",f"{shared:.6f}")
    if args.lambda_sens>0:
        log("Gradient sanity PASSED:",gradient_sanity(model,anchor,colloc))

    hist=[]; start=time.time()
    for imf in range(args.fork_imf+1,args.total_imf+1):
        log(f"IMF {imf}/{args.total_imf} BACKWARD")
        b=model.train_pass_tangent(endpoint["train"],"b",anchor,colloc)
        log(f"IMF {imf}/{args.total_imf} FORWARD")
        f=model.train_pass_tangent(endpoint["train"],"f",anchor,colloc)
        conv=base.convergence(model,endpoint["train"],cfg,device)
        hist.append({"imf":imf,"backward_bridge_loss":b["bridge_loss_last100"],
                     "forward_bridge_loss":f["bridge_loss_last100"],
                     "forward_sensitivity_loss":f["sensitivity_loss_last100"],
                     "train_endpoint_rmse_norm":conv})
        json.dump(hist,open(run_dir/"convergence.json","w"),indent=2)
        torch.save({"model":model.state_dict(),"config":asdict(cfg),"imf":imf,
                    "lambda_sens":args.lambda_sens,"fork_imf":args.fork_imf},
                   run_dir/f"imf_{imf}.pt")
        log(f"IMF {imf} train endpoint RMSE(norm)=",f"{conv:.6f}")

    results={}
    model.net_f.eval(); model.net_b.eval()
    for split in ["test_seen","test_id","test_ood_near","test_ood_far"]:
        results[split]=base.evaluate_split(split,model,endpoint[split],response_eval[split],cfg,device,metadata)

    payload={"seed":args.seed,"lambda_sens":args.lambda_sens,"fork_imf":args.fork_imf,
             "total_imf":args.total_imf,"response_fraction":args.response_fraction,
             "target_mode":args.target_mode,
             "train_seconds":time.time()-start,"metrics":results}
    json.dump(payload,open(run_dir/"metrics.json","w"),indent=2)
    rows=[{"seed":args.seed,"lambda_sens":args.lambda_sens,
           "response_fraction":args.response_fraction,"target_mode":args.target_mode,
           "split":s,**m} for s,m in results.items()]
    with open(run_dir/"metrics.csv","w",newline="") as fp:
        w=csv.DictWriter(fp,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    log("="*80); log("FINAL SUMMARY"); log("="*80)
    for split,m in results.items():
        log(split,"| field_rel",f"{m['field_rel_l2']:.4f}",
            "| Jv_rel",f"{m['directional_j_rel_error']:.4f}",
            "| finite_rel",f"{m['finite_response_rel_l2']:.4f}")

if __name__=="__main__":
    main()
