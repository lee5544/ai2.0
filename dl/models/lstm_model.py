from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .base_model import BaseModel
from .common import pick_config_value, to_bool, to_float, to_int


class LSTMClassifier(BaseModel):
    """
    输入张量形状:
    - [batch, channels, steps]
    内部会转成 [batch, steps, channels]
    """

    arch_name = "lstm"

    def __init__(
        self,
        *,
        in_channels: int,
        num_classes: int,
        lstm_hidden_size: int = 128,
        lstm_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.2,
        hidden_dim: int = 128,
    ) -> None:
        lstm_hidden_size = max(8, int(lstm_hidden_size))
        lstm_layers = max(1, int(lstm_layers))
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes,
            hyperparams={
                "lstm_hidden_size": int(lstm_hidden_size),
                "lstm_layers": int(lstm_layers),
                "bidirectional": bool(bidirectional),
                "dropout": float(dropout),
                "hidden_dim": int(hidden_dim),
            },
        )
        lstm_dropout = float(dropout) if lstm_layers > 1 else 0.0
        self.bidirectional = bool(bidirectional)
        self.lstm = nn.LSTM(
            input_size=int(in_channels),
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=self.bidirectional,
        )
        lstm_output_dim = lstm_hidden_size * (2 if self.bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_output_dim, int(hidden_dim)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=float(dropout)),
            nn.Linear(int(hidden_dim), int(num_classes)),
        )

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        _, (h_n, _) = self.lstm(x)
        if self.bidirectional:
            last_hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        else:
            last_hidden = h_n[-1]
        return self.classifier(last_hidden)


def resolve_lstm_config(*, model_cfg: Any, train_cfg: Any) -> dict[str, Any]:
    hidden_dim = max(8, to_int(pick_config_value(model_cfg, train_cfg, "hidden_dim", default=128), 128))
    return {
        "dropout": to_float(pick_config_value(model_cfg, train_cfg, "dropout", default=0.2), 0.2),
        "hidden_dim": hidden_dim,
        "lstm_hidden_size": max(
            8,
            to_int(
                pick_config_value(model_cfg, train_cfg, "lstm_hidden_size", default=hidden_dim),
                hidden_dim,
            ),
        ),
        "lstm_layers": max(1, to_int(pick_config_value(model_cfg, train_cfg, "lstm_layers", default=2), 2)),
        "lstm_bidirectional": to_bool(
            pick_config_value(
                model_cfg,
                train_cfg,
                "lstm_bidirectional",
                "bidirectional",
                default=True,
            ),
            default=True,
        ),
    }


def build_lstm_model(
    *,
    in_channels: int,
    sequence_length: int,
    num_classes: int,
    config: dict[str, Any],
) -> nn.Module:
    del sequence_length
    return LSTMClassifier(
        in_channels=int(in_channels),
        num_classes=int(num_classes),
        lstm_hidden_size=int(config["lstm_hidden_size"]),
        lstm_layers=int(config["lstm_layers"]),
        bidirectional=bool(config["lstm_bidirectional"]),
        dropout=float(config["dropout"]),
        hidden_dim=int(config["hidden_dim"]),
    )
