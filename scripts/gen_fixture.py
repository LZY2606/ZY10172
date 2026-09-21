#!/usr/bin/env python3
"""生成固定 fixture：fixtures/samples.json
真实模型 Larson-Miller：T_k*(C+log10 t)=b0+b1*s+b2*log10 t
C=20, b0=-3000, b1=46, b2=4800（温域 450–650°C，保证三参数可识别）。
炉次偏置 HA=0、HB=-0.12 dex；噪声 σ=0.10 dex；确定性随机数便于复核。
"""
import json, math, random

C, B0, B1, B2 = 20.0, -3000.0, 46.0, 4800.0
HEAT_BIAS = {"HA": 0.0, "HB": -0.12}
SIGMA_NOISE = 0.10
rng = random.Random(20260922)

def loglife(temp_c, stress, heat, shift=0.0):
    t = temp_c + 273.15
    return (t * C - B0 - B1 * stress) / (B2 - t) + HEAT_BIAS[heat] + shift

def jitter():
    return rng.gauss(0.0, SIGMA_NOISE)

samples = []
def add(sid, heat, temp_c, stress, time_h, status, criterion="rupture",
        load_kind="constant", segments=None, note=""):
    samples.append({
        "id": sid, "heat": heat, "criterion": criterion, "load_kind": load_kind,
        "temp_value": temp_c if load_kind == "constant" else None,
        "temp_unit": "C" if load_kind == "constant" else None,
        "stress_mpa": stress if load_kind == "constant" else None,
        "time_h": round(time_h, 3), "status": status,
        "segments": segments or [], "note": note,
    })

# ---- rupture：已断裂（14 个，450–650°C，跨 HA/HB）----
rupt_design = [
    ("R-001","HA",450, 70), ("R-002","HB",450,110),
    ("R-003","HA",500,100), ("R-004","HB",500,140),
    ("R-005","HA",550,120), ("R-006","HB",550,150),
    ("R-007","HA",550,200), ("R-008","HB",600,180),
    ("R-009","HA",600,220), ("R-010","HB",600,260),
    ("R-011","HA",650,200), ("R-012","HB",650,260),
    ("R-013","HA",650,300), ("R-014","HB",500, 80),
]
for sid, h, tc, ts in rupt_design:
    add(sid, h, tc, ts, 10 ** (loglife(tc, ts, h) + jitter()), "fractured",
        note=f"{tc}°C/{ts}MPa 断裂")

# ---- rupture：右删失（5 个；time_h 仅为寿命下界）----
for sid, h, tc, ts, frac in [
    ("R-015","HA",450, 50,0.50), ("R-016","HA",500, 60,0.45),
    ("R-017","HB",550, 90,0.55), ("R-018","HB",600,150,0.60),
    ("R-019","HA",650,160,0.70),
]:
    ct = 10 ** loglife(tc, ts, h) * frac
    add(sid, h, tc, ts, ct, "censored",
        note=f"截止 {ct:.0f}h 未断，仅为寿命下界")

# ---- 2% 应变准则：独立分组（绝不与 rupture 混拟）----
STRAIN_SHIFT = -0.30
for k, (sid, h, tc, ts, status) in enumerate([
    ("S-001","HA",500,100,"fractured"), ("S-002","HB",500,140,"fractured"),
    ("S-003","HA",550,150,"fractured"), ("S-004","HB",550,200,"censored"),
    ("S-005","HA",600,220,"fractured"), ("S-006","HB",600,260,"fractured"),
]):
    mu = loglife(tc, ts, h, shift=STRAIN_SHIFT)
    t = 10 ** (mu + jitter()) if status == "fractured" else 10 ** mu * 0.55
    add(sid, h, tc, ts, t, status, criterion="strain2pct",
        note="2% 应变准则，独立成组")

# ---- 变载（Robinson 线性累积损伤；仅启用损伤模型时纳入）----
def variable(sid, heat, segs, status, note):
    total = sum(hh for _, _, hh in segs)
    add(sid, heat, None, None, total, status, load_kind="variable",
        segments=[{"temp_value": tc, "temp_unit": "C", "stress_mpa": ss, "hours_h": round(hh,2)}
                  for tc, ss, hh in segs], note=note)

# V-001 HB：550°C/100 运行 800h，再 600°C/200 至断裂
mu1, mu2 = loglife(550,100,"HB"), loglife(600,200,"HB")
last_h = (1.0 - 800.0/10**mu1) * 10 ** (mu2 + jitter())
variable("V-001", "HB", [(550,100,800.0),(600,200,last_h)], "fractured",
         "两段加载，末段断裂；Σt/t_r≈1")
# V-002 HA：500°C/80 运行 1000h，再 550°C/130 运行 500h 后截止未断（删失）
variable("V-002", "HA", [(500,80,1000.0),(550,130,500.0)], "censored",
         "两段加载后截止仍未断，损伤和为下界")

with open("fixtures/samples.json", "w", encoding="utf-8") as f:
    json.dump({"samples": samples}, f, ensure_ascii=False, indent=2)
print(f"wrote {len(samples)} samples")
