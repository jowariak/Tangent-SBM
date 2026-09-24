"""Fixed-case SNS endpoint/spread/finite-response panels, no derivative maps."""
from common import *
METHODS=('conditional','tangent','tsbm')
LABELS=('Physical reference','Conditional DSBM','Tangent-SBM','TSBM')

@torch.no_grad()
def fields(m,record,mean,std,device,seed):
    x=record['x0'][:1].to(device);a=record['a'][:1].to(device);d=record['direction'][:1].to(device)
    base.set_seed(seed);draws=[]
    for _ in range(32):draws.append(m.sample_sde(x,a).cpu()*std+mean)
    ys=torch.stack(draws)
    base.set_seed(seed+50000);changes=[]
    for _ in range(16):
        nb=m._noise_bank(x)
        changes.append((m.sample_sde(x,a+m.cfg.finite_delta*d,noise_bank=nb)-m.sample_sde(x,a,noise_bank=nb)).cpu()*std)
    return dict(mean=ys.mean(0)[0,0],spread=ys.std(0,unbiased=False)[0,0],finite=torch.stack(changes).mean(0)[0,0])

def generate(args):
    dev=base.resolve_device(args.device)
    # Always first stored condition of every split, for every training seed.
    # No metrics or image appearance enter selection.
    for seed in SEEDS:
        for method in METHODS:
            m,mean,std=model(args,method,seed,dev)
            if m.cfg.finite_delta!=.25:raise ValueError('Expected finite delta=.25')
            _,data=final2.load_final2_data(args.data_dir,mean,std)
            for i,split in enumerate(final2.FINAL2_FILES):
                out=args.run_root/f'{seed}_{split}_{method}.pt'
                provenance=dict(seed=seed,method=method,split=split,condition_index=0,endpoint_mc=32,finite_mc=16,finite_delta=.25,
                    sampling_seed=880000+seed*100000+i*1000,checkpoint=sha(checkpoint_path(args,method,seed)),
                    data=sha(args.data_dir/final2.FINAL2_FILES[split]),manifest=sha(args.data_dir/'final2_eval_manifest.json'),
                    source=sha(__file__),base=sha(base.__file__),common=sha(Path(__file__).with_name('common.py')))
                if out.exists():
                    if base.safe_load(out)['provenance']!=provenance:raise ValueError('Stale figure cache')
                    continue
                print(f'Sampling {seed} {split} {method}, fixed condition 0',flush=True)
                r=data[split];true=r['xT_samples_raw'][0]
                reference=dict(mean=true.mean(0)[0],spread=true.std(0,unbiased=False)[0],finite=r['finite_response_raw'][0,0])
                prediction=fields(m,r,mean,std,dev,provenance['sampling_seed'])
                for group in [reference,prediction]:
                    if not all(torch.isfinite(v).all() for v in group.values()):raise ValueError('Nonfinite figure values')
                save(dict(provenance=provenance,reference=reference,prediction=prediction,x0=r['x0_raw'][0,0],a=float(r['a'][0,0]),direction=float(r['direction'][0,0]),true_samples=len(true)),out)
            del m
    render(args)

def draw_page(records,title,png,pdf):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    truth=records[0]['reference'];groups=[truth]+[r['prediction'] for r in records]
    fig,axs=plt.subplots(4,4,figsize=(11,10),layout='constrained')
    for row,key in enumerate(['mean','spread','finite','finite_error']):
        arrays=[(g['finite']-truth['finite']).numpy() if key=='finite_error' else g[key].numpy() for g in groups]
        bound=max(float(np.abs(v).max()) for v in arrays) or 1e-12
        cmap='viridis' if key=='spread' else 'RdBu_r';lo=0 if key=='spread' else -bound
        for col,arr in enumerate(arrays):
            im=axs[row,col].imshow(arr,origin='lower',cmap=cmap,vmin=lo,vmax=bound,interpolation='nearest')
            axs[row,col].set_xticks([]);axs[row,col].set_yticks([])
            if row==0:axs[row,col].set_title(LABELS[col],fontsize=11)
            if col==0:axs[row,col].set_ylabel(['Endpoint mean','Endpoint spread','Finite mean change','Finite-change error'][row])
        fig.colorbar(im,ax=list(axs[row,:]),shrink=.8,pad=.02,label='Raw vorticity units')
    fig.suptitle(title,fontsize=12)
    fig.savefig(png,dpi=220);fig.savefig(pdf);plt.close(fig)

def render(args):
    args.run_root.mkdir(parents=True,exist_ok=True)
    pages=[]
    for seed in SEEDS:
        for split in final2.FINAL2_FILES:
            rs=[base.safe_load(args.run_root/f'{seed}_{split}_{method}.pt') for method in METHODS]
            r=rs[0]
            for other in rs[1:]:
                for key in ['mean','spread','finite']:torch.testing.assert_close(r['reference'][key],other['reference'][key],rtol=0,atol=0)
            name=f'figure_seed_{seed}_{split}'
            title=f"SNS {split} | seed {seed} | stored case 0 | a={r['a']:.3g}, change={.25*r['direction']:.3g}"
            draw_page(rs,title,args.run_root/(name+'.png'),args.run_root/(name+'.pdf'));pages.append(name)
    text='''# SNS qualitative figures (#17)
Fixed condition index 0 from each FINAL2 split, all seeds 32/42/52. Seed32 pages are the predefined main examples; seeds42/52 are companion robustness views. No case selection based on errors or appearance.

Rows show endpoint mean, pointwise endpoint standard deviation, finite mean change at delta=0.25 times the stored intervention direction, and signed finite-change error. Physical reference uses the saved FINAL2 arrays. Model endpoint MC=32; finite-change MC=16 with common noise between nominal and perturbed inputs. The same sampling seeds are used across methods. All values are raw vorticity units. Each row shares one color scale across methods; scales may differ between pages. No derivative maps or training are used.

The reference finite-response target has its own saved Monte Carlo construction; sampling error remains in both targets and model estimates. These descriptive figures accompany the aggregate quantitative tables and do not determine model selection. Exact tensors and checkpoint/data hashes are retained in the .pt caches.
'''
    (args.run_root/'FIGURE_CAPTION.md').write_text(text,encoding='utf-8')
    print('Saved 12 PDF/PNG pages and caption:',args.run_root,flush=True)

def smoke(args):
    dev=base.resolve_device(args.device)
    for method in METHODS:
        m,mean,std=model(args,method,args.seed,dev)
        _,data=final2.load_final2_data(args.data_dir,mean,std)
        x=data['final2_seen']['x0'][:1].to(dev);a=data['final2_seen']['a'][:1].to(dev)
        y=m.sample_sde(x,a)
        if y.shape!=x.shape or not torch.isfinite(y).all():raise ValueError('Invalid sampler')
        del m
    print('PASS: three frozen model loaders, locked FINAL2 data and endpoint samplers.',flush=True)

if __name__=='__main__':
    args=arguments(__doc__,['smoke','generate','render'],'runs/sns_qualitative_v1');globals()[args.mode](args)
