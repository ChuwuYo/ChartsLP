# chartslp 接入 Relentless-main 指引

> 给新 AI 的交接文档：把 chartslp 接入 Relentless-main 这个 Hyperliquid DEX 终端。
> 阅读时间：5 分钟。执行时间：预计 1–3 小时。

## 你是谁，你要做什么

你是一个新 AI，接手的任务是：**让 Relentless-main 项目消费 `chartslp` 库，获得
与 TrendIQ 一致的 29 种图形检测 + 趋势线能力**。

这个 zip 是 `chartslp` 库本身。它原生存在 TrendIQ（trendiq-react）项目的
`packages/chartslp/` 下。现在以独立形式投递到 Relentless-main，让你接入。

## 必读背景（按顺序）

1. **[README.md](README.md)** — chartslp 的对外 API、示例、依赖、测试方式
2. **[CHANGELOG.md](CHANGELOG.md)** — 当前版本：0.1.0
3. **[src/chartslp/types.py](src/chartslp/types.py)** — 公共 TypedDict 契约
4. **本文档** — Relentless 侧接入步骤

## chartslp 能做什么（一分钟速读）

```python
from chartslp import detect_chart_patterns

candles = [
    {"index": 0, "time": 1700000000, "open": 100.0, "high": 101.0,
     "low": 99.0, "close": 100.5, "volume": 1000.0},
    # ... 20 根以上
]

result = detect_chart_patterns(candles, timeframe="1h", trade_type="swing")
# 返回 PatternResult dict，含：
# - primaryPattern: {type, label, bias, confidence, ...}
# - trendLines: [{points: [{time, price}×2], score, touches, ...}]
# - horizontalLevels: [{price, label, touches, ...}]  ← 支撑阻力
# - swingPoints, shapes, stats
```

**29 种 pattern** 涵盖：三角形 / 楔形 / 头肩 / 通道 / 旗形 / 杯柄 / VCP / V 反转 / 趋势结构。

**纯函数**：无 DB / 无网络 / 无 async / 无 FastAPI 耦合。只依赖 numpy。

---

## Relentless-main 现状（你接手前的状态）

### 已有的 pattern 相关代码

`backend/app/core/pattern_detection.py`（134 行）：

- 函数 `detect_candlestick_patterns(candles)`：只检测**烛台形态**（Doji / Hammer /
  Shooting Star / Bullish/Bearish Engulfing / Three White Soldiers / Three Black Crows）
- 这是**单根/三根 K 线的价格行为**，跟 chartslp 做的**结构级图形检测不重叠**
- **关键发现：这个函数目前没有任何消费者**（grep 整个 backend，除了自身定义没有任何
  import / 调用）—— 它是死代码

### 已有的 Support/Resistance 代码

`backend/app/core/support_resistance.py`：基于 pivot point 的支撑阻力检测。
chartslp 的 `horizontalLevels` 输出做类似的事但更精细（含 cluster + flip detection）。
**可能冲突**，见"冲突与取舍"一节。

### 已有的 candle 格式（重要！）

Relentless 用 **Hyperliquid 的压缩 key 格式**：

```python
# Hyperliquid candle
{"t": 1700000000000, "o": "100.0", "h": "101.0", "l": "99.0", "c": "100.5", "v": "1000", ...}
# 字段：t (ms 时间戳), o/h/l/c/v 都是 str 类型
```

chartslp 要求的是：

```python
{"index": int, "time": int(秒), "open": float, "high": float, "low": float, "close": float, "volume": float}
```

**差异**：key 短名 vs 长名、time 毫秒 vs 秒、数值 str vs float、缺 index 字段。
→ **必须写适配器**，见下。

### 已有的 candle 来源

- 存储：`backend/app/services/candle_store.py`（SQLite，WAL 模式）
- 归档：`backend/candle_archiver.py` + `workers/hyperliquid_worker.py`（WS 监听写入）
- API：`backend/app/routers/candles.py` — `GET /api/candles?coin=BTC&interval=1m&start=&end=&limit=`

### 技术栈

- Python 3.12（看 Dockerfile 确认）
- FastAPI + Hyperliquid SDK + websockets
- 依赖用 `requirements.txt`（**不是 uv**）

---

## 整合策略：**并存，不替换**

核心原则：

- chartslp **不替换** `pattern_detection.py`（两者做不同层次的事）
- chartslp **不替换** `support_resistance.py`（但你可以评估是否用 chartslp 的更强版本取代）
- 新增一个**图形检测端点 / 服务**，喂给前端或交易逻辑用

**建议目录**：

```
Relentless-main/
└── backend/
    ├── app/core/
    │   ├── pattern_detection.py      ← 保留（烛台形态，即使死代码）
    │   ├── support_resistance.py     ← 保留
    │   └── chart_patterns.py         ← 新增：chartslp 适配层（你写这个）
    └── requirements.txt              ← 加 chartslp 依赖
```

---

## 接入步骤

### 第 1 步：安装 chartslp

**方案 A：本地 path**（适合 chartslp 跟 Relentless 放一起）

把这个 zip 解压到 `Relentless-main/packages/chartslp/`，然后：

```bash
# Relentless-main/backend/requirements.txt 末尾加
-e ../packages/chartslp
```

缺点：`pip install -r requirements.txt` 的 `-e` 支持有限，某些部署场景（Docker build
时 `packages/` 不在 build context）会失败。

**方案 B：git URL**（推荐）

私仓 [ChuwuYo/ChartsLP](https://github.com/ChuwuYo/ChartsLP) 已有 v0.1.0 tag。
在 Relentless-main 根目录 clone 或用 git URL：

```bash
# requirements.txt 末尾
chartslp @ git+https://github.com/ChuwuYo/ChartsLP.git@v0.1.0
```

私仓需要 GitHub PAT（Fine-grained，只授 ChartsLP 仓库的 `Contents: Read-only`）。
部署：服务器设 `GITHUB_TOKEN` 环境变量或 `~/.netrc`。

**方案 C：直接拷贝源码**（最简但无法升级）

把 `src/chartslp/` 整个拷进 `Relentless-main/backend/app/vendor/chartslp/`。
在消费处 `from app.vendor.chartslp import ...`。不推荐，失去版本化。

**推荐：方案 B**。

### 第 2 步：写 candle 格式适配器

创建 `Relentless-main/backend/app/core/chart_patterns.py`：

```python
"""Adapter between Hyperliquid candles and chartslp.

Hyperliquid candle format (compressed keys, ms timestamp, string values):
    {"t": 1700000000000, "o": "100.0", "h": "101.0", "l": "99.0",
     "c": "100.5", "v": "1000", "T": ..., "i": "1m", "s": "BTC"}

chartslp candle format (standard keys, second timestamp, float values, indexed):
    {"index": 0, "time": 1700000000, "open": 100.0, ...}
"""

from __future__ import annotations

from typing import Any

from chartslp import detect_chart_patterns, PATTERN_LABELS
from chartslp.types import PatternResult


def _normalize_candle(candle: dict, index: int) -> dict:
    """Coerce Hyperliquid candle → chartslp candle."""
    # Hyperliquid uses single-char keys + string values + ms timestamps
    if "c" in candle and "close" not in candle:
        return {
            "index": index,
            "time": int(candle["t"]) // 1000,  # ms → s
            "open": float(candle["o"]),
            "high": float(candle["h"]),
            "low": float(candle["l"]),
            "close": float(candle["c"]),
            "volume": float(candle.get("v", 0)),
        }
    # Already in standard format — just ensure index and int time
    return {
        "index": index,
        "time": int(candle["time"]),
        "open": float(candle["open"]),
        "high": float(candle["high"]),
        "low": float(candle["low"]),
        "close": float(candle["close"]),
        "volume": float(candle.get("volume", 0)),
    }


def detect_patterns(
    candles: list[dict],
    timeframe: str = "1h",
    trade_type: str = "swing",
) -> PatternResult:
    """Run chartslp on a list of Hyperliquid or standard candles.

    Args:
        candles: at least ~20 candles, either Hyperliquid compressed format
            or standard OHLCV.
        timeframe: one of "1m", "5m", "15m", "30m", "1h", "2h", "4h",
            "1d", "1W", "1M".
        trade_type: "swing" or "scalp" — affects pattern tolerances.

    Returns:
        PatternResult — see chartslp/types.py for the full shape.
    """
    if not candles:
        return _empty_result(timeframe, trade_type)

    normalized = [_normalize_candle(c, i) for i, c in enumerate(candles)]
    return detect_chart_patterns(normalized, timeframe, trade_type)


def _empty_result(timeframe: str, trade_type: str) -> dict:
    """Consistent empty shape when no candles were provided."""
    return {
        "version": 1,
        "timeframe": timeframe,
        "tradeType": trade_type,
        "primaryPattern": {"type": "no_clear_pattern", "label": "", "bias": "neutral"},
        "shapes": [],
        "swingPoints": [],
        "trendLines": [],
        "horizontalLevels": [],
        "levels": {"support": None, "resistance": None},
        "stats": {"pivotCount": 0, "atr": 0.0, "volumeRatio": 0.0},
    }


__all__ = ["detect_patterns", "PATTERN_LABELS"]
```

### 第 3 步：暴露 HTTP 端点

在 `backend/app/routers/` 下新建 `chart_patterns.py`：

```python
"""Chart pattern detection endpoint backed by chartslp."""

from fastapi import APIRouter, Query, HTTPException

from app.core.chart_patterns import detect_patterns
from app.services.candle_store import get_candles

router = APIRouter(prefix="/api", tags=["chart-patterns"])


@router.get("/chart-patterns")
async def get_chart_patterns(
    coin: str = Query(..., description="Coin symbol, e.g. BTC"),
    interval: str = Query("1h", description="1m/5m/15m/30m/1h/4h/1d"),
    trade_type: str = Query("swing", alias="tradeType", regex="^(swing|scalp)$"),
    lookback: int = Query(200, le=500, description="Number of recent candles to scan"),
):
    """Run chartslp on the most recent `lookback` candles from the archive."""
    candles = get_candles(coin, interval, limit=lookback)
    if len(candles) < 20:
        raise HTTPException(
            status_code=400,
            detail=f"Need at least 20 candles; got {len(candles)}",
        )
    return detect_patterns(candles, timeframe=interval, trade_type=trade_type)
```

然后在 `backend/app/main.py` 注册：

```python
from app.routers import chart_patterns
app.include_router(chart_patterns.router)
```

### 第 4 步：冒烟测试

```bash
cd Relentless-main/backend
pip install -r requirements.txt
python -c "from chartslp import detect_chart_patterns; print('OK')"
```

启动服务器后：

```bash
curl 'http://localhost:8000/api/chart-patterns?coin=BTC&interval=1h&lookback=200' | jq .primaryPattern
```

预期看到类似 `{"type": "ascending_channel", "label": "Ascending Channel", "bias": "bullish", ...}`。

### 第 5 步：前端渲染（如果需要）

chartslp 输出的 `trendLines` / `shapes` / `horizontalLevels` 是坐标（`{time, price}`
数组）。Relentless 前端用 `lightweight-charts`，可以：

- `trendLines` → 用 `createTrendLine` 或 `addLineSeries` 画
- `horizontalLevels` → `addPriceLine`
- `shapes` → 多边形叠层（需自定义 canvas）

具体怎么接前端看 `Relentless-main/src/components/Chart/` 的现有渲染代码。本 zip 不
包含前端代码。

---

## 冲突与取舍

### 1. chartslp 的 `horizontalLevels` vs Relentless 的 `support_resistance.py`

两者都算支撑阻力，但：

- `support_resistance.py` 的实现是一次性 pivot 扫描，简单
- chartslp 的 `_compute_clustered_levels` 做 K-means-like 聚合 + flip detection（支撑翻阻力）

**建议**：保留 `support_resistance.py` 现有消费者，**新端点用 chartslp 的 `horizontalLevels`**。
等观察一段时间后评估是否统一。

### 2. chartslp 的 pattern vs Relentless 的 `pattern_detection.py`

- `pattern_detection.py` = 烛台形态（单根/三根 K 线）
- chartslp = 图形结构（几十根 K 线）

**不冲突**。都可以保留。如果前端要显示烛台形态（Doji/Hammer），继续用
`pattern_detection.py`；显示图形（Triangle/H&S）用 chartslp。

### 3. 性能

chartslp 对 200 根 K 线的检测大约 50–200ms（CPU，纯 numpy）。
不适合 WS tick 级实时更新，**适合**按蜡烛收盘触发 + HTTP 请求驱动。

如果要 streaming：加一个节流层，比如每 10 秒 / 每根新蜡烛触发一次。

---

## 已知坑位

### 1. candle 字段缺 `volume`

Hyperliquid candle 的 `v` 是浮点字符串，但某些时段 / 某些对 Hyperliquid 不上报成交量，
可能是 `"0"` 或缺失。`_normalize_candle` 里 `candle.get("v", 0)` 已兜底。
但 `stats.volumeRatio` 会因此失真（不影响 pattern 识别本身）。

### 2. 时间戳单位

Hyperliquid K 线 `t` 是**毫秒**，chartslp 要**秒**。转换时 `int(t) // 1000`。
**写测试时用已知的候选时间戳验证**，不要抽象 magic number。

### 3. timeframe 字符串

chartslp 只认这几个：`"1m", "5m", "15m", "30m", "1h", "2h", "4h", "1d", "1W", "1M"`。
Hyperliquid 支持 `"3m"`, `"8h"` 等 chartslp 不认的值 —— 这些会走 chartslp 默认
timeframe 预算，检测质量略降但不会崩。

### 4. 微币种精度

chartslp 的 `smart_round_price` 支持到 1e-9（PEPE / SHIB 级）。Hyperliquid 现货
pair 可能更小（某些 meme spot）。
已测试到 1e-9 没问题，更小的需要补测。

### 5. Docker build context

若用方案 A（本地 path），Dockerfile 的 build context 必须包含 `packages/chartslp/`：

```dockerfile
# 在 Relentless-main 根目录 build
COPY packages/ /app/packages/
COPY backend/ /app/backend/
WORKDIR /app/backend
RUN pip install -r requirements.txt
```

否则 `-e ../packages/chartslp` 找不到源码。**方案 B（git URL）无此坑**。

### 6. chartslp 的公共 API 冻结保证

当前是 v0.1.0，API 未承诺稳定。升级到 v0.2.0 前读 CHANGELOG。
消费时用精确 tag（`@v0.1.0`）钉死，不要用 `main` 分支。

---

## 验收清单

执行完成前确认：

- [ ] `pip install -r requirements.txt` 成功（无报错）
- [ ] `python -c "from chartslp import detect_chart_patterns; print('OK')"` 输出 `OK`
- [ ] `python -c "import chart_patterns" from backend.app.core` 成功（自己的适配器）
- [ ] 启动 FastAPI 服务器，`GET /api/chart-patterns?coin=BTC&interval=1h` 返回 200
- [ ] 响应体里 `primaryPattern.type` 不是 `no_clear_pattern`（对 BTC 1h 足够长的历史）
- [ ] 响应体里 `trendLines` 至少 1 条，`horizontalLevels` 至少 1 条
- [ ] 已有的 `pattern_detection.py` / `support_resistance.py` / 其他功能未被破坏
- [ ] Docker build 成功（如果用容器部署）

## 开发纪律（来自 TrendIQ 同一套规则）

- Python 函数签名全加类型注解
- `async-first`：并发 I/O 用 `asyncio.gather`
- 外部 API 调用用 `httpx.AsyncClient` + 显式 timeout
- 禁止 `except Exception: pass` 静默吞异常
- API 密钥走环境变量或 settings 模块，禁硬编码
- 不要改 chartslp 源码（视为只读的第三方库，真要改到上游 [ChuwuYo/ChartsLP](https://github.com/ChuwuYo/ChartsLP) 提 PR）

---

## 需要回报给人类的事

接入过程中遇到以下情况**立刻停下问人类**：

1. Docker build context 不支持 `packages/` 同步拷入 → 需要改部署架构
2. 私仓 token 拿不到 / SSH key 未配置 → 需要凭证
3. 前端要渲染 `shapes`（多边形）但现有 chart 库不支持 → 需要评估
4. Hyperliquid candle 实际格式跟上面描述不符 → 适配器要改
5. chartslp 检测结果**明显错误**（BTC 1h 明显上升趋势识别成 descending_channel）
   → 可能是适配器 bug 或 chartslp bug，需要定位

顺利完成不用汇报，跑完验收清单直接提交。

---

## 参考

- chartslp 源码：[src/chartslp/](src/chartslp/)
- chartslp 测试：[tests/](tests/)（26 个 pass，是你的"本应正常"基准）
- TrendIQ 的 api 消费示例：原项目 `api/src/api/providers/ai/service.py` 里有完整的
  `detect_chart_patterns + format_pattern_engine_context` 使用模式，注入 LLM prompt
- GitHub：https://github.com/ChuwuYo/ChartsLP（私仓，v0.1.0 tag）
