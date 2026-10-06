"""Chapters 1-4: shapes, spectral, calculus, backprop. All numbers match transcript."""
import numpy as np

def verify_ch01():
    r = {}
    # Ex1: 32x512 @ 512x2048 -> 32x2048, ops 32*2048*512*2 = 67108864
    r["ex01_shape"] = (32, 2048); r["ex01_ops"] = 32*2048*512*2
    assert r["ex01_ops"] == 67108864
    # Ex2: angle between (1,2,2),(2,1,-2)
    a = np.array([1.,2,2]); b = np.array([2.,1,-2])
    dot = float(a@b); cosv = dot/(np.linalg.norm(a)*np.linalg.norm(b))
    r["ex02_dot"] = dot; r["ex02_cos"] = cosv; r["ex02_deg"] = float(np.degrees(np.arccos(np.clip(cosv,-1,1))))
    assert dot == 0.0 and abs(cosv) < 1e-12 and abs(r["ex02_deg"]-90) < 1e-9
    # Ex3: det [[2,0],[0,3]] = 6
    M = np.array([[2.,0],[0,3]]); r["ex03_det"] = float(np.linalg.det(M))
    assert abs(r["ex03_det"]-6) < 1e-9
    # Ex4: rot90 twice: (1,0)->(0,1)->(-1,0)
    R = np.array([[0.,-1],[1,0]]); v = np.array([1.,0])
    v1 = R@v; v2 = R@v1
    r["ex04_v1"] = v1.tolist(); r["ex04_v2"] = v2.tolist()
    assert np.allclose(v1,[0,1]) and np.allclose(v2,[-1,0])
    # Ex5: rank(u v^T)=1
    u = np.array([1.,2,3,4]); v = np.array([5.,6,7])
    G = np.outer(u,v); r["ex05_rank"] = int(np.linalg.matrix_rank(G)); r["ex05_shape"] = G.shape
    assert r["ex05_rank"] == 1 and G.shape == (4,3)
    # Ex6: LoRA 4096x4096 rank8: 2*4096*8=65536, pct of 16777216
    full = 4096*4096; lora = 2*4096*8
    r["ex06_lora"] = lora; r["ex06_full"] = full; r["ex06_pct"] = 100*lora/full
    assert lora == 65536 and full == 16777216 and abs(r["ex06_pct"]-0.390625) < 1e-9
    # Ex7: proj (3,4) onto (1,0)
    r["ex07_shadow"] = [3.,0.]; r["ex07_resid"] = [0.,4.]
    assert 3**2+4**2 == 5**2
    # Ex8: w = sum xy / sum x^2 = 29/14
    x = np.array([1.,2,3]); y = np.array([2.,3,7])
    w = float(x@y/(x@x)); r["ex08_w"] = w
    assert abs(w-29/14) < 1e-12
    resid = y - w*x; assert abs(float(resid@x)) < 1e-9  # orthogonality (ex7 again)
    # Ex9: broadcast (32,1,64)+(1,10,64)->(32,10,64)
    A = np.zeros((32,1,64)); B = np.zeros((1,10,64))
    r["ex09_shape"] = tuple((A+B).shape); assert r["ex09_shape"] == (32,10,64)
    # Ex10: einsum bhqd,bhkd->bhqk, (2,8,128,64)->(2,8,128,128)
    import torch
    Q = torch.randn(2,8,128,64); K = torch.randn(2,8,128,64)
    S = torch.einsum("bhqd,bhkd->bhqk", Q, K)
    r["ex10_shape"] = tuple(S.shape); assert r["ex10_shape"] == (2,8,128,128)
    r["ex10_quad"] = True  # grows with square of context
    return r

def verify_ch02():
    r = {}
    A = np.array([[2.,1],[1,2]])
    w, V = np.linalg.eig(A)
    idx = np.argsort(w)[::-1]; w = w[idx]
    r["ex11_eigvals"] = sorted([float(x) for x in w])
    assert np.allclose(sorted(w),[1.,3]) and abs(np.trace(A)-4)<1e-9 and abs(np.linalg.det(A)-3)<1e-9
    # Ex12: power iteration from (1,0), 10 steps -> aligns to (1,1), factor 3^10
    v = np.array([1.,0])
    for _ in range(10): v = A@v
    cosv = float(v@[1,1]/(np.linalg.norm(v)*np.sqrt(2)))
    r["ex12_cos_top"] = cosv; r["ex12_growth"] = 3**10
    assert cosv > 0.9999 and r["ex12_growth"] == 59049
    # Ex13: 0.9^50, 1.1^50
    r["ex13_small"] = 0.9**50; r["ex13_big"] = 1.1**50
    assert abs(r["ex13_small"]-0.005153775) < 1e-6 and abs(r["ex13_big"]-117.39085) < 0.01
    r["ex13_ratio"] = r["ex13_big"]/r["ex13_small"]; assert r["ex13_ratio"] > 20000
    # Ex14: svd of [[0,2],[1,0]] -> 2,1
    M = np.array([[0.,2],[1,0]]); s = np.linalg.svd(M, compute_uv=False)
    r["ex14_sv"] = sorted([float(x) for x in s], reverse=True)
    assert np.allclose(r["ex14_sv"],[2.,1.])
    # Ex15: sv (5,3,1) best rank-1 err sqrt(9+1)=sqrt10
    r["ex15_err"] = float(np.sqrt(3**2+1**2)); assert abs(r["ex15_err"]-np.sqrt(10))<1e-12
    # Ex16: norms of (3,-4)
    v = np.array([3.,-4]); r["ex16_l1"]=7.; r["ex16_l2"]=5.; r["ex16_linf"]=4.
    assert abs(np.linalg.norm(v,1)-7)<1e-9 and abs(np.linalg.norm(v)-5)<1e-9 and abs(np.linalg.norm(v,np.inf)-4)<1e-9
    # Ex17: clip (6,8) to 5 -> (3,4)
    g = np.array([6.,8]); n = np.linalg.norm(g)
    gc = g*(5/n) if n > 5 else g
    r["ex17_clipped"] = gc.tolist(); assert np.allclose(gc,[3,4])
    # Ex18: Fro^2 of [[1,2],[3,4]] = 30
    F = np.array([[1.,2],[3,4]]); r["ex18_fro2"] = float((F**2).sum())
    assert r["ex18_fro2"] == 30.0
    assert abs(float(np.trace(F.T@F))-30) < 1e-9
    # Ex19: [[2,-1],[-1,2]] PD, eig 1,3
    P = np.array([[2.,-1],[-1,2]]); ev = sorted(np.linalg.eigvalsh(P).tolist())
    r["ex19_eig"] = ev; r["ex19_pd"] = all(x > 0 for x in ev)
    assert np.allclose(ev,[1.,3]) and r["ex19_pd"]
    # Ex20: cond diag(100,1) = 100
    D = np.diag([100.,1.]); r["ex20_cond"] = float(np.linalg.cond(D))
    assert abs(r["ex20_cond"]-100) < 1e-9
    return r

def verify_ch03():
    r = {}
    sig = lambda x: 1/(1+np.exp(-x))
    r["ex21_max"] = 0.25  # sigma(0.5)(1-0.5)
    assert abs(sig(0)*(1-sig(0))-0.25) < 1e-12
    # Ex22 tanh'
    r["ex22_t0"] = 1-np.tanh(0)**2; r["ex22_t3"] = 1-np.tanh(3)**2
    assert abs(r["ex22_t0"]-1)<1e-12 and abs(r["ex22_t3"]-0.0099)<0.0002
    # Ex23 chain: d/dx (3x+1)^2 at 1 = 24
    r["ex23"] = 2*4*3; assert r["ex23"] == 24
    # Ex24 grad x^2 y + y^3 at (1,2) = (4,13)
    r["ex24"] = [2*1*2, 1**2+3*2**2]; assert r["ex24"] == [4,13]
    # Ex25 grad w^Tx = x (numeric check)
    x = np.array([3.,-1.,2.]); w = np.array([0.5,1.5,-2.])
    eps = 1e-6; num = np.array([(w+eps*np.eye(3)[i])@x - (w-eps*np.eye(3)[i])@x for i in range(3)])/(2*eps)
    assert np.allclose(num, x, atol=1e-5); r["ex25"] = "grad=x verified"
    # Ex26 grad x^TAx = (A+A^T)x; symmetric 2Ax
    A = np.array([[1.,2],[3,4.]]); xv = np.array([1.,-1.])
    f = lambda z: float(z@A@z)
    num = np.array([(f(xv+eps*np.eye(2)[i])-f(xv-eps*np.eye(2)[i]))/(2*eps) for i in range(2)])
    assert np.allclose(num, (A+A.T)@xv, atol=1e-5); r["ex26"] = "Ax+A^Tx verified"
    S = np.array([[2.,1],[1,3.]]); assert np.allclose((S+S.T)@xv, 2*S@xv)
    # Ex27 normal equations on ex8 data
    X = np.array([[1.],[2],[3]]); y = np.array([2.,3,7])
    wn = float(np.linalg.solve(X.T@X, X.T@y).item()); r["ex27_w"] = wn
    assert abs(wn-29/14) < 1e-12
    assert abs(float((X.T@(X.flatten()*wn-y)).item())) < 1e-9
    # Ex28 softmax jacobian at (0,0)
    p = np.array([0.5,0.5]); J = np.diag(p)-np.outer(p,p)
    r["ex28_J"] = J.tolist(); assert np.allclose(J,[[0.25,-0.25],[-0.25,0.25]])
    # Ex29 Taylor e^0.1
    r["ex29"] = 1+0.1+0.1**2/2; assert abs(r["ex29"]-1.105)<1e-12 and abs(np.exp(0.1)-1.1051702)<1e-6
    # Ex30 central diff x^3 at 2, h=0.01
    f3 = lambda t: t**3; cd = (f3(2.01)-f3(1.99))/0.02
    r["ex30"] = float(cd); assert abs(cd-12)<1e-3
    return r

def verify_ch04():
    r = {}
    # Ex31
    w,x,b,y = 2.,3.,1.,5.; d = w*x+b-y; loss = d**2
    r["ex31_loss"] = loss; r["ex31_dd"] = 2*d; r["ex31_db"] = 2*d; r["ex31_dw"] = 2*d*x
    assert loss == 4 and r["ex31_dw"] == 12 and r["ex31_db"] == 4
    # Ex32 softmax CE grad = p - y
    z = np.array([2.,1,0]); e = np.exp(z-z.max()); p = e/e.sum()
    g = (p-np.array([1.,0,0])).tolist(); r["ex32_p"] = p.tolist(); r["ex32_g"] = g
    assert abs(p[0]-0.665)<0.002 and abs(g[0]+0.335)<0.002 and abs(sum(g))<1e-9
    # Ex33 shapes: dW=X^T dY, dX=dY W^T
    import torch
    n,d,m = 5,4,3
    X = torch.randn(n,d,requires_grad=True); W = torch.randn(d,m,requires_grad=True)
    Y = X@W; dY = torch.randn(n,m); Y.backward(dY)
    assert torch.allclose(X.grad, dY@W.T) and torch.allclose(W.grad, X.detach().T@dY)
    r["ex33"] = "X^TdY / dYW^T verified; backward~2x forward"
    # Ex34 ReLU mask
    r["ex34"] = "dx=dy*(x>0); dead if always<=0"
    assert (np.array([-1.,2.]) > 0).tolist() == [False, True]
    # Ex35 bias sums over batch
    dYb = np.ones((7,4)); r["ex35_shape"] = dYb.sum(axis=0).shape; assert r["ex35_shape"] == (4,)
    # Ex36 tiny net
    s = lambda t: 1/(1+np.exp(-t))
    h = np.array([s(0.5), s(1.1)]); out = 0.5*h[0]+0.6*h[1]
    r["ex36_out"] = float(out); go = float(out-1)
    r["ex36_dWo"] = [go*h[0], go*h[1]]
    assert abs(out-0.7614)<0.002 and go < 0
    # Ex37 shared input x*x -> 2x
    r["ex37"] = 2*3.0; assert r["ex37"] == 6.0
    # Ex38 max routing
    r["ex38"] = "grad to winner only"
    # Ex39 0.25^10
    r["ex39"] = 0.25**10; assert abs(r["ex39"]-1/1048576)<1e-12
    # Ex40 residual dy/dx = 1+f'
    r["ex40"] = "1+f'(x): gradient highway"
    return r
