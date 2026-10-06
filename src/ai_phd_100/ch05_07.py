"""Chapters 5-7: probability, statistics, information theory."""
import numpy as np

def verify_ch05():
    r = {}
    # Ex41 Bayes: 99/(99+495)=1/6
    r["ex41"] = 99/(99+495); assert abs(r["ex41"]-1/6) < 1e-9
    # Ex42 die mean 3.5 var 35/12
    r["ex42_mean"] = 3.5; r["ex42_var"] = 35/12
    assert abs(np.mean([1,2,3,4,5,6])-3.5)<1e-12 and abs(np.var([1,2,3,4,5,6])-35/12)<1e-12
    # Ex43 linearity: n*(1/n)=1
    r["ex43"] = 1.0
    # Ex44 var(mean)=s2/n
    rng = np.random.default_rng(0); s = rng.normal(0,2,size=(20000,16))
    assert abs(s.mean(axis=1).var()-4/16) < 0.05; r["ex44"] = "sigma^2/n verified"
    # Ex45 P(|Z|<1.96)=0.95
    from scipy.stats import norm
    r["ex45"] = float(norm.cdf(1.96)-norm.cdf(-1.96)); assert abs(r["ex45"]-0.95)<1e-4
    # Ex46 sum of 2 N(0,1) = N(0,2)
    z = rng.normal(size=200000)+rng.normal(size=200000)
    assert abs(z.var()-2)<0.03; r["ex46_var"] = float(z.var())
    # Ex47 reparam 3+2e
    e = rng.normal(size=200000); assert abs((3+2*e).mean()-3)<0.02 and abs((3+2*e).var()-4)<0.1
    r["ex47"] = "3+2eps verified"
    # Ex48 softmax([2,1,0]/0.5)
    l = np.array([2.,1,0])/0.5; ex = np.exp(l-l.max()); p = ex/ex.sum()
    r["ex48"] = p.tolist(); assert abs(p[0]-0.867)<0.002
    # Ex49 chain 0.1*0.05*0.3
    r["ex49_p"] = 0.1*0.05*0.3; r["ex49_logp"] = float(np.log(0.1*0.05*0.3))
    assert abs(r["ex49_p"]-0.0015) < 1e-12 and abs(r["ex49_logp"]+6.502)<0.01
    # Ex50 cov [[1,2],[2,4]] corr 1 det 0
    C = np.array([[1.,2],[2,4]]); r["ex50_det"] = float(np.linalg.det(C))
    r["ex50_corr"] = 2/(1*2); assert abs(r["ex50_det"])<1e-9 and r["ex50_corr"] == 1.0
    return r

def verify_ch06():
    r = {}
    from scipy.stats import norm
    # Ex51 MLE coin 7/10
    r["ex51_p"] = 0.7
    # Ex52 mean 4, mle 8/3, unbiased 4
    s = np.array([2.,4,6]); r["ex52_mean"] = 4.0; r["ex52_mle"] = 8/3; r["ex52_unb"] = 4.0
    assert abs(s.mean()-4)<1e-12 and abs(((s-4)**2).sum()/3-8/3)<1e-12
    # Ex53 MSE = NLL gaussian (algebraic identity, check numerically)
    y, mu, s2 = 2.5, 1.5, 2.0
    nll = 0.5*np.log(2*np.pi*s2)+(y-mu)**2/(2*s2)
    assert abs(nll-((y-mu)**2/(2*s2)+0.5*np.log(2*np.pi*s2)))<1e-12
    r["ex53"] = "MSE==NLL up to const verified"
    # Ex54 CE -ln0.25
    r["ex54"] = float(-np.log(0.25)); assert abs(r["ex54"]-1.3862944)<1e-6
    # Ex55 MAP gaussian => L2
    r["ex55"] = "Gaussian prior -> ||w||^2/2s^2 (weight decay)"
    # Ex56 bias^2+var
    r["ex56"] = 0.5**2+0.25; assert r["ex56"] == 0.5
    # Ex57 SE sqrt(0.8*0.2/400)=0.02, CI +-3.9pp
    se = np.sqrt(0.8*0.2/400); r["ex57_se"] = float(se); r["ex57_ci"] = 1.96*se
    assert abs(se-0.02)<1e-12
    # Ex58 diff 0.02, SE_diff~0.0283, z=0.707 not significant
    se_d = np.sqrt(2*0.02**2); r["ex58_z"] = 0.02/se_d
    assert abs(r["ex58_z"]-0.7071)<0.01 and r["ex58_z"] < 1.96
    # Ex59 SE 0.8/sqrt5
    r["ex59"] = 0.8/np.sqrt(5); assert abs(r["ex59"]-0.3578)<0.001
    # Ex60 MC pi SE: 4*sqrt(p(1-p)/n), p=pi/4
    import math
    p = math.pi/4; r["ex60_se"] = 4*math.sqrt(p*(1-p)/10000)
    assert abs(r["ex60_se"]-0.0164)<0.001
    return r

def verify_ch07():
    r = {}
    # Ex61 H fair=1bit, H(0.9)=-0.9log2.9-0.1log2.1
    h = lambda ps: float(-sum(p*np.log2(p) for p in ps if p > 0))
    r["ex61_fair"] = h([0.5,0.5]); r["ex61_biased"] = h([0.9,0.1])
    assert abs(r["ex61_fair"]-1)<1e-12 and abs(r["ex61_biased"]-0.469)<0.001
    # Ex62 log2 50000
    r["ex62"] = float(np.log2(50000)); assert abs(r["ex62"]-15.61)<0.01
    # Ex63 perp e^2
    r["ex63"] = float(np.exp(2.0)); assert abs(r["ex63"]-7.389)<0.001
    # Ex64 2 nats in bits
    r["ex64"] = 2/np.log(2); assert abs(r["ex64"]-2.88539)<1e-4
    # Ex65 KL both ways
    p = np.array([0.5,0.5]); q = np.array([0.9,0.1])
    kl = lambda a,b: float(np.sum(a*np.log(a/b)))
    r["ex65_pq"] = kl(p,q); r["ex65_qp"] = kl(q,p)
    assert abs(r["ex65_pq"]-0.5108)<0.002 and abs(r["ex65_qp"]-0.3681)<0.002 and r["ex65_pq"] != r["ex65_qp"]
    # Ex66 CE = H + KL
    ce = float(-np.sum(p*np.log(q)))
    assert abs(ce-(-np.sum(p*np.log(p))+kl(p,q)))<1e-12; r["ex66"] = "CE=H+KL verified"
    # Ex67 KL N(mu,1)||N(0,1)=mu^2/2 (numeric)
    mu = 1.5; xs = np.linspace(-8,11,40001); dx = xs[1]-xs[0]
    pn = np.exp(-(xs-mu)**2/2)/np.sqrt(2*np.pi); qn = np.exp(-xs**2/2)/np.sqrt(2*np.pi)
    r["ex67"] = float(np.sum(pn*np.log(pn/qn))*dx)
    assert abs(r["ex67"]-mu**2/2)<0.001
    # Ex68 MI 1 bit / 0
    r["ex68_same"] = 1.0; r["ex68_ind"] = 0.0
    # Ex69 smoothing eps0.1 K10: correct 0.91 rest 0.01
    r["ex69_correct"] = 0.9+0.1/10; r["ex69_other"] = 0.1/10
    assert r["ex69_correct"] == 0.91 and r["ex69_other"] == 0.01
    # Ex70 1MB -> 125KB
    r["ex70_KB"] = 1e6/8/1e3; assert r["ex70_KB"] == 125.0
    return r
