from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch


@dataclass
class Episode:
    """Store all relevant information of an episode.

    The GRPO adaptation for visual models requires us to track richer metadata
    than the original text-only version.  ``group_id`` identifies the input that
    produced a group of Monte Carlo samples, while ``response_tensor`` stores
    the raw tensor response sampled from the model when operating on
    non-generative architectures.  All attributes now provide defaults so
    existing positional instantiations continue to work while the extended
    tensor metadata remains opt-in.
    """

    group_id: Any = None
    reward: float = 0.0
    reward_info: Dict[str, float] = field(default_factory=dict)
    is_finished: bool = True
    prefix: Optional[str] = None
    text: Optional[str] = None
    prefix_token_ids: Optional[List[int]] = None
    prefix_tokens: Optional[List[str]] = None
    generated_token_ids: Optional[List[int]] = None
    response_tensor: Optional[torch.Tensor] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MiniBatch:
    """Batch of data for each training step."""

    prefix: List[str]
    prefix_tokens: List[List[str]]
    prefix_token_ids: List[List[int]]
    numbers: List[List[int]]
    target: List[int]


@dataclass
class TensorMiniBatch:
    """Batch container for tensor-only (e.g. vision) models.

    ``inputs`` should already be stacked along the batch dimension.  Additional
    per-sample metadata can be supplied through ``metadata`` which is useful for
    tasks such as detection where auxiliary annotations are required when
    computing rewards.
    """

    inputs: torch.Tensor
    targets: Optional[Any] = None
    group_ids: Optional[List[Any]] = None
    metadata: Optional[List[Dict[str, Any]]] = None
