import math
from itertools import product
import numpy as np

S0 = 100.0
u = 1.2
d = 0.9
r = 0.05
B = {t: (1 + r) ** t for t in range(3)}
q = ((1 + r) - d) / (u - d)

def stock_price_at_path(path):
    S = S0
    for m in path:
        S *= (u if m == "u" else d)
    return S

paths_t0 = [()]
paths_t1 = list(product("ud", repeat=1))
paths_t2 = list(product("ud", repeat=2))

nodes = {}
nodes[(0, '')] = S0
for p in paths_t1:
    nodes[(1, ''.join(p))] = stock_price_at_path(p)
for p in paths_t2:
    nodes[(2, ''.join(p))] = stock_price_at_path(p)

print("Risk-neutral q =", q)
print("Discounted prices (S/B):")
for (t, path), S in sorted(nodes.items()):
    print(f"t={t}, path={path or '∅'}, S={S:.2f}, B={B[t]:.5f}, S_tilde={S/B[t]:.5f}")

# Martingale checks at t=0 and t=1
def S_tilde(t, path):
    return nodes[(t, path)]/B[t]

for t, path in [(0, ''), (1, 'u'), (1, 'd')]:
    if t == 0:
        ups = S_tilde(1, 'u')
        dns = S_tilde(1, 'd')
    else:
        ups = S_tilde(2, path+'u')
        dns = S_tilde(2, path+'d')
    EQ = q*ups + (1-q)*dns
    print(f"Martingale @ node (t={t}, path={path or '∅'}): E_Q[next S~]={EQ:.5f}, current S~={S_tilde(t, path):.5f}, diff={EQ - S_tilde(t, path):.6f}")

# European call K=100 at T=2
K = 100.0
payoffs = {p: max(nodes[(2, p)] - K, 0.0) for p in ['uu','ud','du','dd']}
probs = {'uu': q**2, 'ud': q*(1-q), 'du': q*(1-q), 'dd': (1-q)**2}
price_by_Q = sum(payoffs[p]*probs[p] for p in payoffs)/B[2]
print("Call price by risk-neutral expectation:", round(price_by_Q, 6))

# Backward induction
def C1(path1):
    up = payoffs[path1+'u']
    dn = payoffs[path1+'d']
    return (q*up + (1-q)*dn)/(1+r)
C0 = (q*C1('u') + (1-q)*C1('d'))/(1+r)
print("Call price by backward induction:", round(C0, 6))
