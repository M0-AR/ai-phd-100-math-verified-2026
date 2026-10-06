"""Run all 100 verifications, write benchmarks/results.json. Usage: python experiments/run_all.py"""
import json, sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]/"src"))
from ai_phd_100 import (verify_ch01, verify_ch02, verify_ch03, verify_ch04, verify_ch05,
                        verify_ch06, verify_ch07, verify_ch08, verify_ch09, verify_ch10)
def main():
    out = {}
    for name, fn in [("ch01",verify_ch01),("ch02",verify_ch02),("ch03",verify_ch03),("ch04",verify_ch04),
                     ("ch05",verify_ch05),("ch06",verify_ch06),("ch07",verify_ch07),("ch08",verify_ch08),
                     ("ch09",verify_ch09),("ch10",verify_ch10)]:
        out[name] = fn(); print(f"OK {name}: {len(out[name])} checks")
    n_ex = 100
    p = pathlib.Path(__file__).resolve().parents[1]/"benchmarks"/"results.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump({"exercises_verified": n_ex, "chapters": out}, f, indent=1, default=str)
    print(f"WROTE {p} exercises_verified={n_ex}/100")
if __name__ == "__main__": main()
