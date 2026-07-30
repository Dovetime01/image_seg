# 工程图纸信息块分割（Info-Block Segmentation）

面向机械 / 铸造等 **工程图纸** 的轻量信息区域分割方案：无需预先框选矩形，即可把视图、标题栏、尺寸注记、比例文字等「信息块」自动切开，并支持人工审核与导出。

适合 CAD 审图、图纸结构化预处理、区域裁剪入库等场景。

---

## 能解决什么问题？

工程图纸上往往没有干净的矩形包围盒：视图与文字靠得很近、膨胀后容易粘连，标题栏与图框线又会「吞掉」整页。本项目针对这类难点提供：

| 痛点 | 本项目做法 |
|------|------------|
| 没有现成矩形框，难以按区域处理 | 二值化 → 近邻膨胀 → 连通域，直接得到信息块 |
| 字与字、字与视图间距不一，一刀切不准 | 可调膨胀核（区域边距）；可选「文字二次膨胀」专门合并碎字 |
| 外框线把整页粘成一块 | 边框清理 / 边框线剥离，保留内部图元 |
| 自动结果仍有粘连或误切 | Web 端手动 **拆分 / 融合 / 删除** 审核 |
| 需要给下游用裁剪件或叠色效果图 | 一键导出叠色 PNG、各组件 ZIP |

> 算法核心是经典视觉管线（OpenCV 形态学 + 连通域），可选 OCR 辅助文字合并；**不依赖大模型推理**，本地即可跑通。

---

## 功能一览

### 自动分割

- 图纸二值化与可选去线（全局 / 仅边框长线）
- 外页边框带清理，避免标题栏被框线吞并
- **膨胀核 `gap_thres`**：控制「多远的笔划仍算同一块」（相对 1200px 参考宽度自动缩放）
- 连通域提取信息块，叠色可视化（半透明区域色 + 描边）
- **文字二次膨胀**：对文字类区域用更大核再合并，减少注记被切碎

### Web 审核 Demo

- 批量导入 **PDF / PNG / JPG**（多页 PDF 自动拆页）
- 分页切换；参数按页保存
- Canvas：**平移**、**拆分**（笔画即边界，大块保色、小块新色）、**融合**、**删除**、撤销
- 查看中间过程图（二值化、膨胀等）
- 导出当前页叠色效果图、各信息块裁剪 ZIP

### CLI / Python API

- 单图调试、`graph/` 批处理、零件裁剪导出
- 可直接 `from segm import extract_info_blocks` 接入业务流水线

---

## 快速开始

### 环境

- Python 3.10+
- Node.js 18+（仅 Web Demo 需要）
- PDF 推荐：`pip install pymupdf`（无需系统 poppler）

### Web Demo（推荐同事试用）

```bash
# 终端 1 — 后端
pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000

# 终端 2 — 前端
cd frontend && npm install && npm run dev
```

浏览器打开：<http://127.0.0.1:5173>

更细的操作说明见 [README_DEMO.md](README_DEMO.md)。

### CLI 单图

```bash
python scripts/demo.py \
  --image graph/resource/某图.png \
  --gap-thres 8 \
  --output-dir info_block_debug/current
```

### Python API

```python
import cv2
from segm import ExtractConfig, extract_info_blocks, draw_region_overlay

img = cv2.imread("graph/resource/135-铸造.png")
cfg = ExtractConfig(gap_thres=8, text_ocr_refine=False)
blocks, debug = extract_info_blocks(img, config=cfg)
overlay = draw_region_overlay(img, debug["label_map"], blocks=blocks, alpha=0.4)
cv2.imwrite("overlay.png", overlay)
```

---

## 仓库结构

```
image_seg/
  segm/          # 分割算法（extract / visualize / text_refine / export）
  scripts/       # CLI 批处理与调试脚本
  graph/         # 样例输入与批处理输出
  backend/       # FastAPI（分析、审核编辑、导出）
  frontend/      # Vite + Vue3 + Canvas 审核界面
  requirements.txt
```

---

## 主要参数

| 参数 | 作用 |
|------|------|
| `gap_thres` | 膨胀核 / 区域边距：笔划相距多远仍合并 |
| `text_ocr_refine` | 文字二次膨胀开关（勾选后需重新分析） |
| `text_gap_thres` | 文字专用更大膨胀核 |
| `line_removal_mode` | `none` / `global` / `border_frame` |
| `border_clear_*` | 清理最外圈图框，避免整页粘连 |
| `alpha` | 叠色透明度 |

调试时输出目录下常见文件：`*_02_binary.png`、`*_05_dilated.png`、`*_08_region_overlay.png`、`*_meta.txt` 等。

---

## 算法流程（简图）

```text
图纸 → 二值化 →（可选去线 / 清边框）→ 膨胀(gap)
     → 连通域 →（可选文字二次膨胀）→ 信息块列表
     → 叠色预览 / 人工拆分·融合 / 导出
```

---

## 许可与说明

本仓库为工程图纸信息块分割的研究 / 演示实现，便于本地试用与二次集成。若用于生产，请结合贵司图纸规范做参数标定与质检流程。

问题与建议欢迎提 Issue / PR。
