#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""黑流树海节点识别（桌面版）：亮/暗双模板 → 光斑过滤 → 垂直网格
用法:
  python3 node_grid.py <截图路径> [输出路径]
  例: python3 node_grid.py C:/Users/Lenovo/Desktop/ak_screenshots/rogue/xxx.png
"""
import glob
import os
import sys

import cv2
import numpy as np

# 素材库（识别文件夹，亮/暗 1-7）
MAT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "识别素材")
MAT_DIR = os.path.normpath(MAT_DIR)
# 检测区域 mask（同目录 检测区域mask.png）
ZONE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "检测区域mask.png")

SCALES = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
NMS_DIST = 45
GAP = 80
GRID_TOL = 70
BIG_GAP = 1.8


def load_zone(path, shape):
    """加载检测区域 mask（PNG 或 npy），按截图尺寸缩放"""
    if not path or not os.path.exists(path):
        return None
    if path.endswith(".npy"):
        z = np.load(path)
    else:
        z = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        z = (z > 100).astype(np.uint8) * 255
    if z.shape != shape:
        z = cv2.resize(z, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
    return z


def template_candidates(img, zone, thresh):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    cands = []
    for f in sorted(glob.glob(os.path.join(MAT_DIR, "*.png")) + glob.glob(os.path.join(MAT_DIR, "亮", "*.png")) + glob.glob(os.path.join(MAT_DIR, "暗", "*.png"))):
        name = os.path.basename(f).split(".")[0]
        # 亮/暗目录标记（激活态=亮，静默态=暗）——同名亮暗素材必须区分
        parent = os.path.basename(os.path.dirname(f))
        if parent in ("亮", "暗"):
            name = f"{parent}/{name}"
        t = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        if t is None:
            continue
        th, tw = t.shape[:2]
        for scale in SCALES:
            ts = cv2.resize(t, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            if ts.shape[0] >= h or ts.shape[1] >= w:
                continue
            res = cv2.matchTemplate(gray, ts, cv2.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= thresh)
            for x, y in zip(xs, ys):
                cx, cy = int(x + tw * scale / 2), int(y + th * scale / 2)
                if zone is None or (0 <= cy < zone.shape[0] and 0 <= cx < zone.shape[1] and zone[cy, cx]):
                    cands.append((cx, cy, float(res[y, x]), name))
    kept = []
    for c in sorted(cands, key=lambda c: -c[2]):
        if all(np.hypot(c[0] - k[0], c[1] - k[1]) > NMS_DIST for k in kept):
            kept.append((c[0], c[1], c[2], c[3]))
    return kept


def is_spot(cx, cy, img, lap_th=200, edge_th=0.02):
    """背景光斑过滤器：光斑是模糊色块（低清晰度+无边缘），真节点有清晰结构"""
    h, w = img.shape[:2]
    if cx < 26 or cy < 26 or cx > w - 26 or cy > h - 26:
        return False
    p = img[cy - 25:cy + 25, cx - 25:cx + 25]
    gray = cv2.cvtColor(p, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    edge = (cv2.Canny(gray, 50, 150) > 0).mean()
    return lap < lap_th and edge < edge_th


def cluster_1d(vals, gap=GAP):
    groups = []
    for v in sorted(vals):
        if groups and abs(v - groups[-1][-1]) <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(np.mean(g)) for g in groups]


def interpolate(axis, cands, other_axis, axis_idx=0, tol=GRID_TOL, big=BIG_GAP):
    """大间距内插 + 边缘外推（axis_idx=0 列用 x，1 排用 y）"""
    if len(axis) < 2:
        return axis
    gaps = [axis[i + 1] - axis[i] for i in range(len(axis) - 1)]
    med = max(40, int(np.median(gaps)))
    out = list(axis)
    for probe in (axis[0] - med, axis[-1] + med):
        if probe < 0 or probe > 2000:
            continue
        if axis_idx == 0:
            hits = [c for c in cands if abs(c[0] - probe) <= tol
                    and any(abs(c[1] - o) <= tol for o in other_axis)]
        else:
            hits = [c for c in cands if abs(c[1] - probe) <= tol
                    and any(abs(c[0] - o) <= tol for o in other_axis)]
        # 外推：≥2 候选 或 1 个高分候选(≥0.75) 才成立（防低分误检假列，不漏单节点真列如"你的位置"）
        if len(hits) >= 2 or (len(hits) == 1 and hits[0][2] >= 0.75):
            out.append(probe)
    out.sort()
    added = True
    while added:
        added = False
        for i in range(len(out) - 1):
            d = out[i + 1] - out[i]
            if d > big * med:
                n_ins = int(round(d / med)) - 1
                for k in range(1, n_ins + 1):
                    probe = out[i] + med * k
                    if axis_idx == 0:
                        hits = [c for c in cands if abs(c[0] - probe) <= tol
                                and any(abs(c[1] - o) <= tol for o in other_axis)]
                    else:
                        hits = [c for c in cands if abs(c[1] - probe) <= tol
                                and any(abs(c[0] - o) <= tol for o in other_axis)]
                    if len(hits) >= 1:
                        out.append(probe)
                        added = True
                out.sort()
                break
    return out


def fit_grid(hi, lo):
    col_x = cluster_1d([c[0] for c in hi])
    row_y = cluster_1d([c[1] for c in hi])
    col_x = interpolate(col_x, hi + lo, row_y, axis_idx=0)
    row_y = interpolate(row_y, hi + lo, col_x, axis_idx=1)
    return col_x, row_y


POS_TOL = 25  # 匹配位置偏差容差：真匹配偏差≤6px，跑偏>25px 是文字区/相邻内容的假匹配


def build_nodes(col_x, row_y, hi, lo):
    nodes = []
    for ci, cx in enumerate(col_x):
        for ri, ry in enumerate(row_y):
            hh = [c for c in hi if abs(c[0] - cx) <= GRID_TOL and abs(c[1] - ry) <= GRID_TOL]
            ll = [c for c in lo if abs(c[0] - cx) <= GRID_TOL and abs(c[1] - ry) <= GRID_TOL]
            # 位置偏差过滤：只采信匹配位置靠近网格交叉点的候选（真匹配偏差≤6px）
            hh = [c for c in hh if abs(c[0] - cx) <= POS_TOL and abs(c[1] - ry) <= POS_TOL]
            ll = [c for c in ll if abs(c[0] - cx) <= POS_TOL and abs(c[1] - ry) <= POS_TOL]
            if hh:
                b = max(hh, key=lambda c: c[2])
                nodes.append((ci, ri, cx, ry, True, b[2], b[3], "高"))
            elif ll:
                b = max(ll, key=lambda c: c[2])
                nodes.append((ci, ri, cx, ry, True, b[2], b[3], "低"))
            else:
                nodes.append((ci, ri, cx, ry, False, 0, "", "空"))
    return nodes


def main():
    if len(sys.argv) < 2:
        print("用法: python3 node_grid.py <截图路径> [输出路径]")
        sys.exit(1)
    shot = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.splitext(shot)[0] + "-网格.png"

    base = cv2.imread(shot)
    if base is None:
        print(f"❌ 读不到图片: {shot}")
        sys.exit(1)
    h, w = base.shape[:2]
    zone = load_zone(ZONE_PATH, base.shape[:2])

    hi = template_candidates(base, zone, 0.80)
    lo = template_candidates(base, zone, 0.60)
    hi = [c for c in hi if not is_spot(c[0], c[1], base)]
    lo = [c for c in lo if not is_spot(c[0], c[1], base)]
    print(f"素材库: {MAT_DIR}")
    print(f"高阈值候选 {len(hi)} 个, 低阈值候选 {len(lo)} 个（已滤光斑）")

    col_x, row_y = fit_grid(hi, lo)
    print(f"网格: {len(col_x)} 列 {col_x} × {len(row_y)} 排 {row_y}")
    nodes = build_nodes(col_x, row_y, hi, lo)
    n_node = sum(1 for n in nodes if n[4])
    print(f"交叉点: {len(nodes)} 个（节点 {n_node}，空格 {len(nodes) - n_node}）")

    vis = base.copy()
    for ci, cx in enumerate(col_x):
        cv2.line(vis, (cx, row_y[0]), (cx, row_y[-1]), (100, 255, 100), 1)
    for ri, ry in enumerate(row_y):
        cv2.line(vis, (col_x[0], ry), (col_x[-1], ry), (100, 255, 100), 1)
    for ci, ri, x, y, is_node, s, name, src in nodes:
        if is_node:
            col = (0, 255, 255) if src == "高" else (255, 170, 0)
            cv2.circle(vis, (x, y), 10, col, 2)
            cv2.putText(vis, f"({ci},{ri})", (x + 13, y - 7),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 1)
        else:
            cv2.circle(vis, (x, y), 8, (120, 120, 120), 1)
            cv2.putText(vis, f"({ci},{ri})?", (x + 10, y + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1)
    if zone is not None:
        overlay = np.zeros((h, w, 3), np.uint8)
        overlay[zone > 0] = (0, 200, 0)
        vis = cv2.addWeighted(vis, 1.0, overlay, 0.05, 0)
    cv2.imwrite(out, vis)
    print(f"✅ 结果图: {out}")


if __name__ == "__main__":
    main()
