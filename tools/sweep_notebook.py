"""Smoke-test a notebook headlessly, exercising every interactive control.

Each `widgets.interact(f, **controls)` call is replaced by calls to `f` with the default value of every control,
then with every other option of each dropdown, both ends of each slider, and each checkbox flipped. Figures are
closed instead of shown (their PDFs are still written to figures/). Run from the repository root inside the
QIIME 2 environment:

    python tools/sweep_notebook.py class_nasal_microbiome.ipynb
    python tools/sweep_notebook.py biom200_microbiome.ipynb --set 'TREE = None'

Exits non-zero on the first error.
"""
import argparse
import json
import sys
import time
import traceback

STUB = '''
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as _plt
_plt.show = lambda *a, **k: _plt.close("all")
import ipywidgets as _w


def _sweep(f, **kw):
    base = {k: v.value for k, v in kw.items()}
    combos = [dict(base)]
    for k, v in kw.items():
        if isinstance(v, _w.Dropdown):
            combos += [{**base, k: o} for o in v.options if o != base[k]]
        elif isinstance(v, _w.IntSlider):
            combos += [{**base, k: v.min}, {**base, k: v.max}]
        elif isinstance(v, _w.FloatLogSlider):
            combos += [{**base, k: v.base ** v.min}, {**base, k: v.base ** v.max}]
        elif isinstance(v, _w.Checkbox):
            combos += [{**base, k: not base[k]}]
    for c in combos:
        f(**c)
    print(f"    swept {f.__name__}: {len(combos)} combinations OK", flush=True)


_w.interact = _sweep
import scripts.biom200 as _m
_m.widgets.interact = _sweep
'''


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("notebook")
    ap.add_argument("--set", action="append", default=[], metavar="'NAME = value'",
                    help="replace the line assigning NAME in the notebook's code (repeatable)")
    a = ap.parse_args()
    sys.path.insert(0, ".")
    cells = [c for c in json.load(open(a.notebook))["cells"] if c["cell_type"] == "code"]
    env = {"__name__": "__main__"}
    exec(STUB, env)
    t0 = time.time()
    for i, c in enumerate(cells):
        src = "".join(c["source"])
        src = "\n".join(l for l in src.splitlines() if not l.startswith("%"))
        for s in a.set:
            name = s.split("=")[0].strip()
            src = "\n".join(s if l.split("=")[0].strip() == name and "=" in l else l for l in src.splitlines())
        first = src.strip().splitlines()[0][:70] if src.strip() else ""
        t = time.time()
        try:
            out = eval(compile(src, f"cell{i}", "eval"), env) if "\n" not in src.strip() and "=" not in src and src.strip() \
                else exec(compile(src, f"cell{i}", "exec"), env)
        except SyntaxError:
            out = exec(compile(src, f"cell{i}", "exec"), env)
        except Exception:
            traceback.print_exc()
            print(f"FAILED in code cell {i}: {first}")
            sys.exit(1)
        if out is not None and hasattr(out, "to_string"):
            print(out.to_string(max_rows=20))
        elif out is not None and hasattr(out, "data"):   # pandas Styler
            print(out.data.to_string(max_rows=20))
        print(f"{time.time() - t:6.1f}s  {first}", flush=True)
    print(f"ALL CELLS OK in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
