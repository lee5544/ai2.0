"""卡片插件接口 + 注册表。

每个分析 = backend/cards/ 下一个独立 py 文件，定义 Card 子类并用 @register 注册。
卡片只负责"算" → 返回 Plotly figure JSON；前端用通用 Plotly.js 渲染器渲染，
因此新增分析只需丢一个 py 文件，无需改前端。
"""
from __future__ import annotations

import importlib
import json
import pkgutil
from dataclasses import dataclass

import plotly.graph_objs as go

from .tdms_loader import SampleSignals


@dataclass
class CardContext:
    """注入给卡片的只读上下文（单通道：sample_id 已含 up/down）。"""
    index: int
    sn: str
    sample_id: str
    direction: str
    line: str
    sr: int
    raw: object       # np.ndarray | None
    proc: object      # np.ndarray | None

    @classmethod
    def from_signals(cls, sig: SampleSignals) -> "CardContext":
        return cls(index=sig.index, sn=sig.sn, sample_id=sig.sample_id,
                   direction=sig.direction, line=sig.line, sr=sig.sampling_rate,
                   raw=sig.raw, proc=sig.proc)


class Card:
    id: str = ""
    title: str = ""
    category: str = "spectrum"   # waveform / spectrum / feature / task
    default: bool = False        # 是否默认出现在通道面板
    order: int = 100             # 显示排序（越小越靠前）
    # 可调参数声明，前端据此生成控件。每项：
    #   {key, label, type:"number"|"select", default, options?, min?, max?, step?}
    params: list[dict] = []

    def build(self, ctx: CardContext, p: dict) -> go.Figure:
        raise NotImplementedError

    def _merge_defaults(self, params: dict | None) -> dict:
        out = {spec["key"]: spec.get("default") for spec in self.params}
        if params:
            for k, v in params.items():
                if k in out and v is not None and v != "":
                    out[k] = v
        return out

    # 统一出口：算图 -> Plotly JSON dict
    def render(self, ctx: CardContext, params: dict | None = None) -> dict:
        p = self._merge_defaults(params)
        fig = self.build(ctx, p)
        figure = json.loads(fig.to_json()) if fig is not None else None
        return {"id": self.id, "title": self.title, "category": self.category,
                "params": self.params, "used_params": p, "figure": figure}


REGISTRY: "dict[str, Card]" = {}


def register(cls):
    inst = cls()
    if not inst.id:
        raise ValueError(f"Card {cls.__name__} 缺少 id")
    REGISTRY[inst.id] = inst
    return cls


def discover() -> None:
    """扫描 backend.cards 包，导入全部卡片模块（触发 @register）。"""
    from . import cards as cards_pkg
    for m in pkgutil.iter_modules(cards_pkg.__path__):
        importlib.import_module(f"{cards_pkg.__name__}.{m.name}")


def _sorted_cards() -> list[Card]:
    return sorted(REGISTRY.values(), key=lambda c: (c.order, c.id))


def list_cards() -> list[dict]:
    return [{"id": c.id, "title": c.title, "category": c.category,
             "default": c.default, "order": c.order, "params": c.params}
            for c in _sorted_cards()]


def default_card_ids() -> list[str]:
    return [c.id for c in _sorted_cards() if c.default]


def render_card(card_id: str, ctx: CardContext, params: dict | None = None) -> dict:
    card = REGISTRY.get(card_id)
    if card is None:
        raise KeyError(f"未知卡片: {card_id}")
    return card.render(ctx, params)
