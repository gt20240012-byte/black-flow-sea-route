#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""判路 v40.6（终版 + 亮节点 + 险路宽松处理 + 密度过滤）：密度主段 + 模板高阈值兜底
规则：路 = (密度主段占比≥0.35 且像素≥30) 或 (模板相关度≥0.80 且像素≥30)
- 密度主段：40-bin 剖面最大连续有效段占比；有效 = bin≥8像素（大哥密度法：
  真路一整条持续高密度，特效星星点点每bin只有1-3像素不算数）
- 模板兜底：高阈值 0.80 只救弱路，挡住光污染误判
- v40.1 新增：亮节点检测（V>180 占比>2% = 可行动/你的位置），输出 bright_nodes
- v40.2 新增：险路尽头/险路恶敌节点特效排除——排除其周围60px圆运动像素，
  险路边要求主段≥0.45+延伸段≥8（真路贯穿整边，特效弥散带只有0.28-0.40悬空在中段）
- v40.3 新增：bin密度阈值8（有效bin才算主段），星星点点特效被打回原形
- v40.4/40.5 修正：险路分类处理——险路尽头宽松判定 corr≥0.75（误杀真路修复，12:27局大哥确认）
- v40.6 修正：险路恶敌也用宽松判定！120529“弥散假路”结论有误（只看排除后分布，
  排除区恰好清了真路的节点端运动）；最新局证实恶敌真路排除前紧贴节点有密集运动
验证：v40 基础 59/59 + 7x5 险路局（v40.2 3条污染边全删）+ 全局边无回归
用法: python3 判路_v34b.py <视频> <截图> <输出前缀> [模板文件]
"""
import sys, os, glob, json
import cv2
import numpy as np

# 自包含：所有资源相对脚本所在目录（一条龙版文件夹可整体拷走）
BASE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)
import node_grid as ng

VID, SHOT, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
TPL_FILE = sys.argv[4] if len(sys.argv) > 4 else os.path.join(BASE, "road_templates.json")
MAT_DIR = os.path.join(BASE, "识别素材")   # 亮/暗 素材库（自包含）
TEXT_TPL_DIR = os.path.join(BASE, "text_templates")

TH_PIX = 30    # 最小运动像素
TH_RUN = 0.35  # 密度主段占比阈值
TH_CORR = 0.80 # 模板相关度兜底阈值（弱路）
TH_TEXT = 0.85  # 文字采信阈值（真实匹配≥0.87，林间空地等无文字节点误匹配≤0.74）
BIN_DENSITY = 8 # 有效bin密度阈值（大哥密度法）：路=持续高密度带(每bin≥8px)，星星点点特效(<8px)不算数

# ---------- 1. 网格 ----------
img = cv2.imread(SHOT)
# 素材完整性检查（发布版不含素材，需用户自备）
_has_mats = (glob.glob(os.path.join(MAT_DIR, "亮", "*.png"))
             + glob.glob(os.path.join(MAT_DIR, "暗", "*.png")))
if not _has_mats:
    print("[错误] 识别素材库为空（识别素材/亮 与 识别素材/暗 文件夹）！")
    print("       请按 README「自备素材」说明，从游戏截图裁剪节点图标放入对应文件夹。")
    sys.exit(2)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
zone = ng.load_zone(os.path.join(BASE, "检测区域mask.png"), gray.shape)
hi = ng.template_candidates(img, zone, 0.80)
lo = ng.template_candidates(img, zone, 0.60)
hi = [c for c in hi if not ng.is_spot(c[0], c[1], img)]
lo = [c for c in lo if not ng.is_spot(c[0], c[1], img)]
col_x, row_y = ng.fit_grid(hi, lo)
nodes = ng.build_nodes(col_x, row_y, hi, lo)
node_map = {}
node_types = {}   # (ci,ri) -> 素材名（节点类型）
node_scores = {}  # (ci,ri) -> 匹配分
for ci, ri, x, y, is_node, s, name, src in nodes:
    node_map[(ci, ri)] = (int(x), int(y), bool(is_node))
    # 素材变体归一化：林间空地-暗2/暗3 → 林间空地，你的位置-箭头版 → 你的位置
    base = name.replace("-箭头版", "").replace("-暗2", "").replace("-暗3", "").split("_")[0]
    node_types[(ci, ri)] = base
    node_scores[(ci, ri)] = float(s)
print(f"网格 {len(col_x)}x{len(row_y)}")

# 你的位置识别（箭头+文字飘动会污染上方边，精准排除矩形）
YOU_RECT = dict(w=25, h_up=70, h_dn=10)  # 节点上方 50x80 矩形
you_node = None
pos_mats = [f for f in glob.glob(os.path.join(MAT_DIR, "*.png"))
            + glob.glob(os.path.join(MAT_DIR, "亮", "*.png"))
            + glob.glob(os.path.join(MAT_DIR, "暗", "*.png"))
            if "位置" in os.path.basename(f)]
if pos_mats:
    g_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    best_pos = -1
    you_xy = None
    for pm in pos_mats:
        t = cv2.imread(pm, cv2.IMREAD_GRAYSCALE)
        if t is None:
            continue
        th_, tw_ = t.shape[:2]
        for sc in [0.6, 0.8, 1.0, 1.2, 1.4]:
            ts = cv2.resize(t, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            if ts.shape[0] >= g_gray.shape[0] or ts.shape[1] >= g_gray.shape[1]:
                continue
            res = cv2.matchTemplate(g_gray, ts, cv2.TM_CCOEFF_NORMED)
            _, mx, _, ml = cv2.minMaxLoc(res)
            if mx > best_pos:
                best_pos = mx
                you_xy = (ml[0] + tw_*sc/2, ml[1] + th_*sc/2)
    if you_xy is not None and best_pos > 0.6:
        best_d = 1e9
        for (ci, ri), (x, y, isn) in node_map.items():
            if not isn:
                continue
            d = np.hypot(x-you_xy[0], y-you_xy[1])
            if d < best_d:
                best_d = d; you_node = (ci, ri)
        if you_node:
            print(f"你的位置(模板识别): {you_node} 分{best_pos:.2f}，排除上方箭头文字区")

# 大节点检测：模板匹配「大节点」素材（Hough 在光污染下会误检光晕，不可靠）
BIG_MATS = [f for f in glob.glob(os.path.join(MAT_DIR, "*.png"))
            + glob.glob(os.path.join(MAT_DIR, "亮", "*.png"))
            + glob.glob(os.path.join(MAT_DIR, "暗", "*.png"))
            if "大节点" in os.path.basename(f)]
big_nodes = set()
if BIG_MATS:
    g_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    for bm in BIG_MATS:
        t = cv2.imread(bm, cv2.IMREAD_GRAYSCALE)
        if t is None:
            continue
        th_, tw_ = t.shape[:2]
        for sc in [0.6, 0.8, 1.0, 1.2, 1.4]:
            ts = cv2.resize(t, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            if ts.shape[0] >= g_gray.shape[0] or ts.shape[1] >= g_gray.shape[1]:
                continue
            res = cv2.matchTemplate(g_gray, ts, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= 0.6)
            for x_, y_ in zip(xs, ys):
                cx, cy = int(x_ + tw_*sc/2), int(y_ + th_*sc/2)
                best_d = 1e9; best_k = None
                for (ci, ri), (x, y, isn) in node_map.items():
                    if not isn:
                        continue
                    d = np.hypot(x-cx, y-cy)
                    if d < best_d:
                        best_d = d; best_k = (ci, ri)
                if best_k is not None and best_d < 30:
                    big_nodes.add(best_k)
if big_nodes:
    print(f"检测到大节点(模板匹配): {sorted(big_nodes)}，周围边用中段密度标准")

# 亮节点检测（v40.1）：可行动节点/你的位置 = 高亮光晕（V>180 占比>2%），与实托邦识别同判据
BRIGHT_TH = 0.02   # 节点区 V>180 像素占比阈值
NODE_R = 55        # 节点检测半径
g_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
v_ch = g_hsv[:, :, 2]
bright_nodes = set()
bright_frac = {}   # (ci,ri) -> V>180 占比
for (ci, ri), (x, y, isn) in node_map.items():
    pv = (v_ch[y-NODE_R:y+NODE_R, x-NODE_R:x+NODE_R] > 180).mean()
    bright_frac[(ci, ri)] = float(pv)
    if pv > BRIGHT_TH:
        bright_nodes.add((ci, ri))
if bright_nodes:
    print(f"[亮] 亮节点(可行动/你的位置): {sorted(bright_nodes)}")

# ---------- 2. 运动累积图 ----------
cap = cv2.VideoCapture(VID)
fps = cap.get(cv2.CAP_PROP_FPS)
STEP = max(2, int(fps / 2))
prev_gray = prev_grad = None
accum = None
frames_used = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    if idx % STEP != 0:
        continue
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(g, (3, 3), 0)
    grad = cv2.Laplacian(g, cv2.CV_32F)
    if prev_gray is not None:
        diff = cv2.absdiff(grad, prev_grad)
        d8 = cv2.convertScaleAbs(diff)
        _, d8 = cv2.threshold(d8, 6, 255, cv2.THRESH_BINARY)
        accum = d8.astype(np.float32) if accum is None else accum + d8
        frames_used += 1
    prev_gray, prev_grad = g, grad
cap.release()
motion = (accum >= 0.4 * frames_used).astype(np.uint8) * 255
motion = cv2.morphologyEx(motion, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
motion = cv2.morphologyEx(motion, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
nl, labels, stats, _ = cv2.connectedComponentsWithStats(motion, 8)
out_m = np.zeros_like(motion)
for i in range(1, nl):
    if stats[i, cv2.CC_STAT_AREA] >= 30:
        out_m[labels == i] = 255
motion = out_m
print(f"运动像素 {(motion>0).sum()}")

# ---------- 3. 文字模板匹配（节点类型增强验证） ----------
TEXT_TPL_DIR = os.path.join(BASE, "text_templates")
text_tpls = {}
if os.path.isdir(TEXT_TPL_DIR):
    for tf in glob.glob(os.path.join(TEXT_TPL_DIR, "*.png")):
        t = cv2.imread(tf, cv2.IMREAD_GRAYSCALE)
        if t is not None:
            text_tpls[os.path.basename(tf)[:-4]] = t
    print(f"文字模板库 {len(text_tpls)} 个（林间空地等无文字节点靠图标识别）")

def match_text_type(ci, ri):
    """在节点下方 y+5~y+145 全带扫描，多尺度匹配文字模板，返回 (类型, 置信度)"""
    if not text_tpls:
        return "", 0.0
    nx, ny = node_map[(ci, ri)][0], node_map[(ci, ri)][1]
    region = cv2.cvtColor(img[ny+5:ny+65, nx-70:nx+70], cv2.COLOR_BGR2GRAY)
    best = (0.0, "")
    for tname, t in text_tpls.items():
        for sc in [0.7, 0.8, 0.9, 1.0, 1.1, 1.2]:
            ts = cv2.resize(t, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
            if ts.shape[0] >= region.shape[0] or ts.shape[1] >= region.shape[1]:
                continue
            res = cv2.matchTemplate(region, ts, cv2.TM_CCOEFF_NORMED)
            _, mx, _, _ = cv2.minMaxLoc(res)
            if mx > best[0]:
                best = (mx, tname)
    # 去掉模板名的坐标后缀（如 未知的诡秘_2_1 → 未知的诡秘）
    clean = best[1].split("_")[0] if best[1] else ""
    return clean, best[0]

# ---------- 4. 模板 ----------
if os.path.exists(TPL_FILE):
    tpl = json.load(open(TPL_FILE, encoding="utf-8"))
    templates = [np.array(t["profile"]) for t in tpl["templates"]]
    print(f"模板库 {len(templates)} 条")
else:
    templates = []

# 险路节点特效排除（v40.2）：险路尽头/险路恶敌 节点本身有持续动画（漩涡/烟雾），
# 会污染所有连接边的运动特征 → 排除其周围圆形区域再判边（同“你的位置”排除思路）
RISK_R_LOOSE = 40   # 险路尽头：温和特效，排除40px只清节点动画主体（v40.4 从60缩到40）
RISK_R_STRICT = 60  # 险路恶敌：剧烈弥散特效，排除60px
RISK_CORR_LOOSE = 0.75  # 险路尽头相关度阈值（真路(0,1)→(0,2) corr=0.78，假路≤0.65）
RISK_TH = 0.45  # 险路恶敌主段阈值
RISK_NEAR = 8   # 险路恶敌延伸段阈值
risk_loose = set()   # 险路尽头：宽松判定
risk_strict = set()  # 险路恶敌：严格判定
for (ci, ri), t in node_types.items():
    if "险路恶敌" in t:
        risk_strict.add((ci, ri))
    elif "险路尽头" in t:
        risk_loose.add((ci, ri))
for (ci, ri) in node_map:
    if node_map[(ci, ri)][2] and (ci, ri) not in risk_loose and (ci, ri) not in risk_strict:
        txt, ts = match_text_type(ci, ri)
        if "险路恶敌" in txt:
            risk_strict.add((ci, ri))
        elif "险路尽头" in txt:
            risk_loose.add((ci, ri))
risk_nodes = risk_loose | risk_strict
if risk_nodes:
    print(f"!! 险路节点: 尽头(宽松)={sorted(risk_loose)} 恶敌(严格)={sorted(risk_strict)}")

def risk_skip(x, y):
    """是否落在险路节点特效排除区内（尽头40px / 恶敌60px）"""
    for (ci, ri) in risk_loose:
        rx, ry = node_map[(ci, ri)][0], node_map[(ci, ri)][1]
        if (x - rx) ** 2 + (y - ry) ** 2 <= RISK_R_LOOSE ** 2:
            return True
    for (ci, ri) in risk_strict:
        rx, ry = node_map[(ci, ri)][0], node_map[(ci, ri)][1]
        if (x - rx) ** 2 + (y - ry) ** 2 <= RISK_R_STRICT ** 2:
            return True
    return False

def corr(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return -1.0 if na == 0 or nb == 0 else float(np.dot(a, b) / (na * nb))

def edge_feats(a, b):
    pa = node_map[a][:2]; pb = node_map[b][:2]
    L = np.hypot(pb[0]-pa[0], pb[1]-pa[1])
    dx, dy = (pb[0]-pa[0])/L, (pb[1]-pa[1])/L
    nx, ny = -dy, dx
    bins = np.zeros(40)
    for bi in range(40):
        t = (bi+0.5)/40
        px = pa[0]+dx*t*L; py = pa[1]+dy*t*L
        for k in range(-12, 13):
            x = int(round(px+nx*k)); y = int(round(py+ny*k))
            if 0 <= x < motion.shape[1] and 0 <= y < motion.shape[0] and motion[y, x]:
                # 排除"你的位置"上方箭头文字飘动区
                if you_node is not None:
                    yx, yy = node_map[you_node][0], node_map[you_node][1]
                    if yy-YOU_RECT['h_up'] <= y <= yy+YOU_RECT['h_dn'] and abs(x-yx) <= YOU_RECT['w']:
                        continue
                # 排除险路节点特效区（v40.2）
                if risk_skip(x, y):
                    continue
                bins[bi] += 1
    n_pix = bins.sum()
    # 有效bin密度过滤（大哥密度法 v40.3）：星星点点的特效像素(每bin<8px)不算“有”
    eff = bins >= BIN_DENSITY
    mr = 0; cur = 0
    for v in eff:
        cur = cur + 1 if v else 0
        mr = max(mr, cur)
    frac = mr / 40
    # 险路端紧贴段像素：距险路节点 60-95px 的 bin 内运动像素（真路从节点出发延伸）
    near_risk = 0
    if risk_strict:
        # 该边的险路端点（a 或 b 中属于 risk_strict 的）
        for rn in risk_strict:
            if a == rn or b == rn:
                risk_end = rn
                is_a_end = (a == rn)
                break
        else:
            risk_end = None
        if risk_end is not None:
            d_end = L if is_a_end else 0.0  # bin 距险路段端点的参考：a 端则距离从 a 算，b 端从 b 算
            for bi in range(40):
                t = (bi + 0.5) / 40
                d_from_a = t * L
                d_from_risk = d_from_a if is_a_end else (L - d_from_a)
                if 60 <= d_from_risk <= 95:
                    near_risk += bins[bi]
    # 模板相关度（20 bin）
    if n_pix > 0 and templates:
        b20 = np.zeros(20)
        for bi in range(20):
            t = (bi+0.5)/20
            px = pa[0]+dx*t*L; py = pa[1]+dy*t*L
            for k in range(-12, 13):
                x = int(round(px+nx*k)); y = int(round(py+ny*k))
                if 0 <= x < motion.shape[1] and 0 <= y < motion.shape[0] and motion[y, x]:
                    if risk_skip(x, y):
                        continue
                    b20[bi] += 1
        p = b20 / b20.sum()
        c = max(corr(p, t) for t in templates)
    else:
        c = -1.0
    return frac, c, n_pix, L, bins, near_risk

# ---------- 4. 判路 ----------
edges = []
for ci in range(len(col_x)-1):
    for ri in range(len(row_y)):
        if (ci,ri) in node_map and (ci+1,ri) in node_map and node_map[(ci,ri)][2] and node_map[(ci+1,ri)][2]:
            edges.append(((ci,ri),(ci+1,ri)))
for ri in range(len(row_y)-1):
    for ci in range(len(col_x)):
        if (ci,ri) in node_map and (ci,ri+1) in node_map and node_map[(ci,ri)][2] and node_map[(ci,ri+1)][2]:
            edges.append(((ci,ri),(ci,ri+1)))

results = []
for a, b in edges:
    frac, c, n_pix, L, bins, near_risk = edge_feats(a, b)
    big = a in big_nodes or b in big_nodes
    strict = a in risk_strict or b in risk_strict
    loose = a in risk_loose or b in risk_loose
    if strict:
        # 险路恶敌（v40.6 修正）：也用宽松判定！120529“弥散假路”结论有误——
        # 当时只看排除后分布（悬空），没看排除前；最新局证实恶敌真路在排除前紧贴节点有密集运动
        road = bool((frac >= TH_RUN and n_pix >= TH_PIX) or (c >= RISK_CORR_LOOSE and n_pix >= TH_PIX))
        results.append((a, b, road, frac, c, n_pix))
        print(f"边 {a}-{b}: 恶敌(宽松) 主段={frac:.2f} 相关={c:.2f} 像素={n_pix:4.0f} → {'路' if road else '断'}")
    elif loose:
        # 险路尽头（温和特效）：宽松判定，相关≥0.75（真路0.78过，假路≤0.65断）
        road = bool((frac >= TH_RUN and n_pix >= TH_PIX) or (c >= RISK_CORR_LOOSE and n_pix >= TH_PIX))
        results.append((a, b, road, frac, c, n_pix))
        print(f"边 {a}-{b}: 尽头 主段={frac:.2f} 相关={c:.2f} 像素={n_pix:4.0f} → {'路' if road else '断'}")
    elif big:
        # 大节点边：烟从大节点喷出并贯穿，看中段(bin8-31)密度
        mid = bins[8:32].sum()
        road = bool(mid >= 40 and n_pix >= 100)
        results.append((a, b, road, frac, c, n_pix))
        print(f"边 {a}-{b}: 大节点 中段密度={mid:.0f} 像素={n_pix:4.0f} → {'路' if road else '断'}")
    else:
        road = bool((frac >= TH_RUN and n_pix >= TH_PIX) or (c >= TH_CORR and n_pix >= TH_PIX))
        results.append((a, b, road, frac, c, n_pix))
        print(f"边 {a}-{b}: 主段={frac:.2f} 相关={c:.2f} 像素={n_pix:4.0f} → {'路' if road else '断'}")

# ---------- 5. 可视化 ----------
vis = img.copy()
for a, b, road, frac, c, n_pix in results:
    pa = node_map[a][:2]; pb = node_map[b][:2]
    col = (0, 200, 0) if road else (0, 0, 255)
    cv2.line(vis, pa, pb, col, 3)
for (ci, ri), (x, y, is_node) in node_map.items():
    if (ci, ri) in bright_nodes:
        # 亮节点：橙色大光晕圈 + 实心点（可行动/你的位置）
        cv2.circle(vis, (x, y), 18, (0, 165, 255), 3)
        cv2.circle(vis, (x, y), 6, (0, 165, 255), -1)
    if is_node:
        cv2.circle(vis, (x, y), 10, (0, 255, 255), 2)
        tname = node_types.get((ci, ri), "")
        if tname:
            cv2.putText(vis, tname, (x+12, y+25), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)
        cv2.putText(vis, f"({ci},{ri})", (x+12, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
        if (ci, ri) in bright_nodes:
            cv2.putText(vis, "可行动", (x+12, y+45), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1)
    else:
        cv2.circle(vis, (x, y), 8, (120,120,120), 1)
cv2.imwrite(OUT + "_roads.png", vis)
cv2.imwrite(OUT + "_motion.png", motion)
print(f"[OK] 输出: {OUT}_roads.png / {OUT}_motion.png")

# ---------- 6. 路网数据导出（机器用 JSON） ----------
data = {
    "grid": {"cols": col_x, "rows": row_y},
    "you_here": you_node,
    "big_nodes": sorted(big_nodes),
    "bright_nodes": sorted(bright_nodes),   # 亮节点 = 可行动/你的位置（V>180 高亮光晕）
    "nodes": [{"ci": ci, "ri": ri, "x": x, "y": y, "is_node": bool(isn),
               "bright": (ci, ri) in bright_nodes,
               "bright_frac": round(bright_frac.get((ci, ri), 0.0), 4),
               "type": node_types.get((ci, ri), "").replace("-箭头版", ""),
               "score": node_scores.get((ci, ri), 0.0),
               "text": (lambda r: match_text_type(ci, ri))(0) if isn else ("", 0.0)}
              for (ci, ri), (x, y, isn) in sorted(node_map.items())],
    "edges": [{"a": list(a), "b": list(b), "road": road,
               "main_frac": round(frac, 3), "corr": round(c, 3), "pix": int(n_pix)}
              for a, b, road, frac, c, n_pix in sorted(results)],
    "roads": [[list(a), list(b)] for a, b, road, frac, c, n_pix in results if road],
    "version": "v40.6",
}
# 综合节点类型：文字分≥TH_TEXT 采信文字，否则图标
for n in data["nodes"]:
    if n["is_node"]:
        txt, txt_s = n["text"]
        n["type_final"] = txt if txt_s >= TH_TEXT else n["type"]
    else:
        n["type_final"] = ""

json_out = OUT + "_roads.json"
with open(json_out, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=1)
print(f"[OK] 路网数据: {json_out}（{len(data['roads'])} 条路）")
