#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实托邦识别 v3（大哥特征版）：中心密集连续烟雾
核心特征（大哥提供）：实托邦 = 节点中心烟雾密集且连续飘动 + 周围薄雾环
判据：中心区(半径35px)活跃帧占比 > CENTER_ACTIVE_TH 且 非亮节点
- 实托邦：中心活跃 0.96-1.00（每帧都在飘）
- 险路恶敌：0.39（偶发动画）→ 排除
- 可行动/安静：0.00-0.07 → 排除
- 你的位置：0.80 但高亮(V>180) → 亮节点过滤排除
用法: python3 utopia_detect3.py <视频> <截图> <输出前缀>
"""
import sys, os, json, glob
import cv2
import numpy as np

sys.path.insert(0, "/mnt/c/Users/Lenovo/Desktop/黑流树海识别")
import node_grid as ng

VID, SHOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]

BRIGHT_TH = 0.02       # 亮节点：节点区 V>180 占比（可行动/你的位置）
CENTER_ACTIVE_TH = 0.8  # 中心活跃占比（实托邦边缘烟雾略稀薄）
CENTER_R = 35          # 中心区半径
NODE_R = 55

# ---------- 1. 网格 + 节点（含空格） ----------
img = cv2.imread(SHOT)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
v_ch = hsv[:, :, 2]
zone = ng.load_zone("/mnt/c/Users/Lenovo/Desktop/黑流树海识别/检测区域mask.png", gray.shape)
hi = ng.template_candidates(img, zone, 0.80)
lo = ng.template_candidates(img, zone, 0.60)
hi = [c for c in hi if not ng.is_spot(c[0], c[1], img)]
lo = [c for c in lo if not ng.is_spot(c[0], c[1], img)]
col_x, row_y = ng.fit_grid(hi, lo)
nodes = ng.build_nodes(col_x, row_y, hi, lo)
node_pos = {}
for ci, ri, x, y, is_node, s, name, src in nodes:
    node_pos[(ci, ri)] = (int(x), int(y), bool(is_node), name)
print(f"网格 {len(col_x)}x{len(row_y)}，交叉点 {len(node_pos)} 个")

# ---------- 2. 亮节点过滤（静态高亮） ----------
bright = set()
for k, (x, y, isn, name) in node_pos.items():
    pv = (v_ch[y-NODE_R:y+NODE_R, x-NODE_R:x+NODE_R] > 180).mean()
    if pv > BRIGHT_TH:
        bright.add(k)
print(f"亮节点(过滤): {sorted(bright) if bright else '无'}")

# ---------- 3. 中心活跃检测 ----------
cap = cv2.VideoCapture(VID)
prev = None
active = {k: 0 for k in node_pos}
total = 0
while True:
    ok, fr = cap.read()
    if not ok:
        break
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    if prev is not None:
        d = cv2.absdiff(g, prev)
        for k, (x, y, isn, name) in node_pos.items():
            yy, xx = np.ogrid[-CENTER_R:CENTER_R, -CENTER_R:CENTER_R]
            cm = np.sqrt(xx**2 + yy**2) <= CENTER_R
            c_roi = d[y-CENTER_R:y+CENTER_R, x-CENTER_R:x+CENTER_R][cm]
            if (c_roi > 8).sum() > 15:
                active[k] += 1
        total += 1
    prev = g
cap.release()

# ---------- 4. 判定 ----------
print(f"\n{'坐标':<8} {'节点?':<5} {'类型':<10} {'中心活跃':<8} {'类别'}")
utopia = []
for k, (x, y, isn, name) in node_pos.items():
    frac = active[k] / max(1, total)
    if k in bright:
        cat = "🔆 亮节点(过滤)"
    elif frac >= CENTER_ACTIVE_TH:
        cat = "🌫️ 实托邦"
        utopia.append(k)
    elif frac >= 0.3:
        cat = "偶发动画(非实托邦)"
    elif frac > 0:
        cat = "弱运动"
    else:
        cat = "安静"
    print(f"({k[0]},{k[1]})  {'是' if isn else '空格':<5} {name:<10} {frac:<8.2f} {cat}")

print(f"\n✅ 实托邦节点: {sorted(utopia)}")
json.dump({"bright": [list(k) for k in sorted(bright)],
           "utopia": [list(k) for k in sorted(utopia)],
           "center_active": {f"({k[0]},{k[1]})": active[k] / max(1, total) for k in active}},
          open(OUT + "_utopia.json", "w"), ensure_ascii=False, indent=1)
print(f"✅ JSON: {OUT}_utopia.json")

# ---------- 5. 可视化 ----------
vis = img.copy()
for k, (x, y, isn, name) in node_pos.items():
    frac = active[k] / max(1, total)
    if k in bright:
        color = (0, 255, 255)
    elif frac >= CENTER_ACTIVE_TH:
        color = (0, 0, 255)
    elif frac >= 0.3:
        color = (255, 0, 255)
    elif frac > 0:
        color = (0, 165, 255)
    else:
        color = (0, 255, 0)
    cv2.circle(vis, (x, y), NODE_R, color, 4)
    cv2.putText(vis, f"{frac:.0%}", (x - 28, y - NODE_R - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
cv2.imwrite(OUT + "_utopia.png", vis)
print(f"✅ 可视化: {OUT}_utopia.png（红=实托邦 黄=亮 紫=偶发 橙=弱 绿=安静）")

# ========== v3.1 扩展：实托邦运动强度分级（2026-08-07 大哥要求） ==========
def intensity_grades(VID, node_pos, center_r=35, utopia=None):
    """计算各节点中心区运动强度（变化幅度），实托邦内部按强度分级"""
    cap = cv2.VideoCapture(VID)
    prev = None
    inten = {k: [] for k in node_pos}
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        if prev is not None:
            d = cv2.absdiff(g, prev)
            for k, (x, y, isn, name) in node_pos.items():
                inten[k].append(d[y-center_r:y+center_r, x-center_r:x+center_r].mean())
        prev = g
    cap.release()
    grades = {}
    for k in node_pos:
        s = np.array(inten[k])
        mean_i = float(s.mean())
        if utopia and k in utopia:
            if mean_i >= 1.0:
                grade = "强(烟雾浓)"
            elif mean_i >= 0.6:
                grade = "中(烟雾中)"
            else:
                grade = "弱(烟雾薄)"
        else:
            grade = "-"
        grades[k] = (mean_i, grade)
    return grades

if __name__ == "__main__" and "--grades" in sys.argv:
    grades = intensity_grades(VID, node_pos, utopia=set(utopia))
    print("\n=== 运动强度分级（中心区变化幅度）===")
    for k, (x, y, isn, name) in sorted(node_pos.items(), key=lambda v: -grades[v[0]][0]):
        if not isn:
            continue
        mi, g = grades[k]
        print(f"({k[0]},{k[1]})  {name:<10} 强度={mi:.2f}  {g}")
    json.dump({f"({k[0]},{k[1]})": {"intensity": grades[k][0], "grade": grades[k][1]}
               for k in grades}, open(OUT + "_grades.json", "w"), ensure_ascii=False, indent=1)
    print(f"✅ 分级JSON: {OUT}_grades.json")

# ========== v3.2 扩展：实托邦内文字识别（大哥方案：雾不遮字） ==========
TEXT_TPL_DIR = "/mnt/c/Users/Lenovo/Desktop/黑流树海识别/text_templates"

def text_match_type(img, x, y, text_tpls, y0=5, y1=145, th=0.6):
    """文字模板匹配：实托邦的雾是动态的，不遮节点下方名字 → 文字识别可靠"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    roi = gray[y+y0:y+y1, max(0, x-130):x+130]
    best = []
    for name, t in text_tpls.items():
        for sc in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]:
            ts = cv2.resize(t, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            if ts.shape[0] >= roi.shape[0] or ts.shape[1] >= roi.shape[1]:
                continue
            res = cv2.matchTemplate(roi, ts, cv2.TM_CCOEFF_NORMED)
            mx = res.max()
            if mx > th:
                best.append((mx, name))
    if not best:
        return None
    best.sort(reverse=True)
    return best[0]

def load_text_tpls():
    tpls = {}
    if os.path.isdir(TEXT_TPL_DIR):
        for tf in glob.glob(os.path.join(TEXT_TPL_DIR, "*.png")):
            t = cv2.imread(tf, cv2.IMREAD_GRAYSCALE)
            if t is not None:
                tpls[os.path.basename(tf)[:-4].replace("_0_1", "")] = t
    return tpls

if __name__ == "__main__" and "--text" in sys.argv:
    tpls = load_text_tpls()
    print(f"文字模板库 {len(tpls)} 个")
    print("\n=== 实托邦节点文字识别（阈值0.6，识别不出=当没有/绕过去）===")
    text_results = {}
    for k in utopia:
        x, y = node_pos[k][0], node_pos[k][1]
        r = text_match_type(img, x, y, tpls)
        if r:
            text_results[f"({k[0]},{k[1]})"] = {"type": r[1], "score": float(round(r[0], 3))}
            print(f"({k[0]},{k[1]}): {r[1]} (分{r[0]:.2f}) ✅")
        else:
            text_results[f"({k[0]},{k[1]})"] = {"type": None, "score": 0}
            print(f"({k[0]},{k[1]}): 识别不出 → 当没有/绕过去")
    json.dump(text_results, open(OUT + "_text.json", "w"), ensure_ascii=False, indent=1)
    print(f"✅ 文字识别JSON: {OUT}_text.json")
