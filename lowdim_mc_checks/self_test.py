"""CPU checks of batching, native tangent differentiation, and metric accumulation."""
import torch
import evaluate as e


class Linear(torch.nn.Module):
    def forward(self, x, u, t):
        return 0.2*x + 0.3*u.expand_as(x)


for module, dim in [(e.gaussian, 2), (e.double_well, 1)]:
    cfg = module.Config(num_steps=4)
    model = module.ConditionalDSBM(cfg, dim, 1, torch.device('cpu'))
    model.net_f = Linear()
    model.nets['f'] = model.net_f
    x, u = torch.zeros(3, dim), torch.ones(3, 1)
    noise = torch.stack([e.noise_for_draw('test','test_id',32,d,(4,3,dim)) for d in range(2)], dim=1)
    batched = model.tangent_rollout(x.repeat(2,1), u.repeat(2,1), noise.reshape(4,6,dim)).reshape(2,3,dim,1)
    separate = torch.stack([model.tangent_rollout(x,u,noise[:,i]) for i in range(2)])
    torch.testing.assert_close(batched, separate)
    expected = 0.3/0.2*((1+0.2/4)**4-1)
    torch.testing.assert_close(batched, torch.full_like(batched,expected))
    # The exact discrete endpoint for zero noise, and paired finite change
    # for arbitrary shared noise, are known for this linear drift.
    endpoint = e.sample_quantity(model,x,u,torch.zeros_like(noise[:,0]),'endpoint',0.25)
    torch.testing.assert_close(endpoint,torch.full_like(endpoint,expected))
    change = e.sample_quantity(model,x,u,noise[:,0],'finite_change',0.25)
    torch.testing.assert_close(change,torch.full_like(change,0.25*expected))
    repeat_change = e.sample_quantity(model,x.repeat(2,1),u.repeat(2,1),noise.reshape(4,6,dim),'finite_change',0.25)
    torch.testing.assert_close(repeat_change,torch.full_like(repeat_change,0.25*expected))

samples = torch.tensor([[[[1.]]], [[[3.]]]], dtype=torch.float64)
r = e.metrics(samples.sum(0), samples.square().sum(0), 2, torch.ones(1,1,1,dtype=torch.float64))
assert abs(r['E_J']-1)<1e-12 and abs(r['estimated_mc_noise_relative']-1)<1e-12
for kind, name in [('endpoint','mean_rmse'),('finite_change','finite_response_rmse')]:
    r = e.output_metrics(samples.sum(0),samples.square().sum(0),2,torch.ones(1,1,1),kind)
    assert abs(r[name]-1)<1e-12
print('PASS: native tangents, endpoint, paired finite change, batching, and metric formulas.')
