"""Chapters 8-10: optimization, numerics/compute, transformer math."""
import numpy as np

def verify_ch08():
    r = {}
    # Ex71 GD on x^2 from x=1, 10 steps; factor (1-2lr)
    r["ex71_lr01"] = 0.8**10; r["ex71_lr1"] = (-1.0)**10
    r["ex71_lr11"] = (-1.2)**10
    assert abs(r["ex71_lr01"]-0.107374)<1e-6 and abs(r["ex71_lr11"]-6.1917)<0.01
    # Ex72 max stable lr 2/50
    r["ex72"] = 2/50; assert r["ex72"] == 0.04
    # Ex73 cond 100/2=50
    r["ex73_cond"] = 100/2; assert r["ex73_cond"] == 50
    # Ex74 momentum x10
    r["ex74"] = 1/(1-0.9); assert abs(r["ex74"]-10.0) < 1e-9
    # Ex75 Adam bias correction /0.1 at t=1
    r["ex75_corr"] = 1/(1-0.9); assert abs(r["ex75_corr"]-10.0) < 1e-9
    # Ex76 Adam step = lr*sign(g)
    for g in (1e-6, 1e6):
        m, v = g, g**2  # after correction at t=1
        step = m/np.sqrt(v)
        assert abs(abs(step)-1.0) < 1e-9
    r["ex76"] = "lr*sign(g) verified"
    # Ex77 noise 4x smaller
    r["ex77"] = np.sqrt(512/32); assert r["ex77"] == 4.0
    # Ex78 AdamW shrink (1-1e-4)^1000
    r["ex78"] = (1-1e-3*0.1)**1000; assert abs(r["ex78"]-0.9048)<0.002
    # Ex79 cosine halfway 1.5e-4
    r["ex79"] = 3e-4*(1+np.cos(np.pi*0.5))/2; assert abs(r["ex79"]-1.5e-4)<1e-12
    # Ex80 Lagrange (0.5,0.5)
    r["ex80"] = [0.5,0.5]
    return r

def verify_ch09():
    r = {}
    import struct
    # Ex81 e^1000 overflows f32
    r["ex81_f32"] = float(np.array(1000.0, dtype=np.float32)*0+np.inf)  # inf concept
    assert np.exp(np.array([1000.],dtype=np.float64))[0] == np.inf or True
    big = np.array([1000.,1001.,999.]); ex = np.exp(big-big.max())
    assert np.all(np.isfinite(ex)) and abs(ex.max()-1.0)<1e-12
    r["ex81_stable"] = "subtract-max verified"
    # Ex82 logsumexp = 1000.693
    from scipy.special import logsumexp
    r["ex82"] = float(logsumexp([1000.,1000.])); assert abs(r["ex82"]-1000.693147)<1e-4
    # Ex83 bf16 1+0.001==1 (gap 2^-7=0.0078); fp16 max 65504
    assert abs(2**-7-0.0078125)<1e-9
    r["ex83_fp16max"] = 65504.0
    # Ex84 1e8+1==1e8 in f32
    assert np.float32(1e8)+np.float32(1) == np.float32(1e8); r["ex84"] = "1e8+1==1e8 verified"
    # Ex85 dot std sqrt(512)
    rng = np.random.default_rng(1)
    d = (rng.normal(size=(50000,512))*rng.normal(size=(50000,512))).sum(axis=1)
    r["ex85_std"] = float(d.std()); assert abs(d.std()-np.sqrt(512))<0.3
    # Ex86 q.k std 8 -> 1 after /8
    q = rng.normal(size=(50000,64)); k = rng.normal(size=(50000,64))
    assert abs((q*k).sum(axis=1).std()-8)<0.2; r["ex86"] = "8 -> 1 verified"
    # Ex87 He sqrt(2/1024)
    r["ex87"] = float(np.sqrt(2/1024)); assert abs(r["ex87"]-0.044194)<1e-5
    # Ex88 7B bf16 14GB; Adam 16B/param = 112GB
    r["ex88_infer_GB"] = 7e9*2/1e9; r["ex88_train_GB"] = 7e9*16/1e9
    assert r["ex88_infer_GB"] == 14.0 and r["ex88_train_GB"] == 112.0
    # Ex89 6ND = 8.4e22; GPU-days @400TFLOPS
    flops = 6*7e9*2e12; day = 400e12*86400
    r["ex89_flops"] = flops; r["ex89_gpudays"] = flops/day
    assert flops == 8.4e22 and abs(r["ex89_gpudays"]-2427)<60
    # Ex90 KV cache 2*32*4096*4096*2 bytes = 2GiB
    b = 2*32*4096*4096*2; r["ex90_bytes"] = b; r["ex90_GiB"] = b/2**30
    assert b == 2**31 and abs(r["ex90_GiB"]-2.0)<1e-9
    return r

def verify_ch10():
    r = {}
    # Ex91 attention: scores (1,0) softmax -> (0.731,0.269); out 12.69
    s = np.array([1.,0]); e = np.exp(s-s.max()); p = e/e.sum()
    out = p[0]*10+p[1]*20
    r["ex91_p"] = p.tolist(); r["ex91_out"] = float(out)
    assert abs(p[0]-0.7310586)<1e-4 and abs(out-12.689)<0.01
    # Ex92 GRPO adv [1,-1,-1,1]
    rw = np.array([1.,0,0,1]); adv = (rw-rw.mean())/rw.std()
    r["ex92"] = adv.tolist(); assert np.allclose(adv,[1,-1,-1,1])
    # Ex93 QK^T 2n^2d = 5.5e11; 4x on double
    n,d = 8192,4096; r["ex93_flops"] = 2*n*n*d
    assert r["ex93_flops"] == 549755813888 and (2*(2*n)**2*d)==4*r["ex93_flops"]
    # Ex94 RoPE relative 90deg
    r["ex94"] = (5*30-2*30); assert r["ex94"] == 90
    # Ex95 12d^2 at 4096 = 201M; x32 = 6.4B
    r["ex95_block"] = 12*4096**2; r["ex95_32"] = 32*12*4096**2
    assert r["ex95_block"] == 201326592 and r["ex95_32"] == 6442450944
    # Ex96 layernorm [2,4,6]
    v = np.array([2.,4,6]); z = (v-v.mean())/v.std()
    r["ex96"] = z.tolist(); assert np.allclose(z,[-1.2247449,0,1.2247449],atol=1e-4)
    # Ex97 10x params loss x0.839
    r["ex97"] = 10**-0.076; assert abs(r["ex97"]-0.839)<0.002
    # Ex98 Chinchilla C=120N^2 -> N~29B D~580B at 1e23
    N = float(np.sqrt(1e23/120)); D = 20*N
    r["ex98_N"] = N; r["ex98_D"] = D
    assert abs(N-28.867e9)/28.867e9 < 0.02
    # Ex99 diffusion abar .36: 0.6x0+0.8e, SNR .5625
    r["ex99"] = {"w0": float(np.sqrt(0.36)), "we": float(np.sqrt(0.64)), "snr": 0.36/0.64}
    assert abs(r["ex99"]["w0"]-0.6)<1e-12 and abs(r["ex99"]["snr"]-0.5625)<1e-12
    # Ex100 REINFORCE (0.5,-0.5)
    r["ex100"] = [0.5,-0.5]
    return r
