"""L3 JAX port vs numpy simulator (scripts/drone_bucket_l23.py): same instance, same rule -> same result.
Runs in a subprocess with JAX x64 so that float comparisons are exact-ish and other tests are not affected."""
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = os.path.join(ROOT, "scripts")


def _run(code):
    env = dict(os.environ, JAX_ENABLE_X64="True", JAX_PLATFORMS="cpu", PYTHONPATH=SCRIPTS)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT, timeout=900)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


CODE = r'''
import json, numpy as np, jax.numpy as jnp
import drone_l3_jax as J
import drone_bucket_l23 as L23
out = []
cells = [J.Cell(5,"ar",50,0.15,0.2,10,0.0), J.Cell(5,"iid",0,0.3,0.2,1,0.15), J.Cell(5,"none",0,0.0,0.0,10,0.0),
         J.Cell(10,"ar",10,0.3,0.0,1,0.0)]
rules = [("fixed",0,0,0.0),("rot",240,0,0.0),("rot",20,0,0.0),("local",5,5,0.0),("local",10,2,0.05),("local",20,10,0.0),("local",5,2,0.05)]
for cell in cells:
    k,B,w,ev = J.eval_instances(cell, 24)
    k,B = k[:24],B[:24]
    n = len(k)
    w = None if w is None else w[:, :n]
    if ev is not None: ev = (ev[0][:n], ev[1][:n], ev[2][:n])
    env = J.Env(cell)
    mvec,sref,ceff = J.cell_params(cell)
    M = np.tile(mvec,(n,1)); ce = np.full(n, ceff)
    inst = J.inst_from_numpy(k,B,ev)
    for rule,p1,p2,p3 in rules:
        if rule=="fixed": kw=dict(policy="fixed")
        elif rule=="rot": kw=dict(policy="rot",p1=p1)
        else: kw=dict(policy="local",p1=p1,p2=p2,p3=p3,init="identity")
        # numpy: rot/fixed in JSON use init sorted; with identity layout we compare like with like
        kw["init"]="identity"
        ref,ns = L23.simulate(k,B,M,ce,w,ev,**kw)
        got,st = J.run_rule(env, inst, rule, p1, p2, p3, w=w)
        assert not bool(np.asarray(st.alive).any())
        out.append(dict(cell=cell.name, rule=rule, p=[p1,p2,p3], maxdiff=float(np.abs(np.asarray(got)-ref).max()),
                        swaps=float(np.asarray(st.nswap).mean()), ref_mean=float(ref.mean())))
print(json.dumps(out))
'''


def test_jax_matches_numpy_rules():
    out = _run(CODE)
    assert len(out) == 28
    for r in out:
        assert r["maxdiff"] < 1e-9, r
    # sanity: the comparison is not vacuous (rules actually swapped in at least some settings)
    assert any(r["swaps"] > 3 for r in out)
    assert any(r["rule"] == "local" and r["swaps"] > 3 for r in out)
