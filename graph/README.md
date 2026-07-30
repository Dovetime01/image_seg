# graph/

本地样例与批处理结果目录（与 Web Demo 会话内存无关）。

| 子目录 | 用途 |
|--------|------|
| `resource/` | 原始 PDF / PNG 输入 |
| `output/` | 批处理最终叠色图副本 |
| `_render_cache/` | PDF 渲染缓存 |
| `*/N/` | 各图纸按页存放的调试输出 |
| `debug_text/` | 文字二次膨胀对比实验 |

Web Demo 导入文件不会写入这里；要用 CLI 批处理时：

```bash
python scripts/run_graph_segm.py
```
