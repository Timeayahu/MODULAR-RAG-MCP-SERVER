"""Trace 上下文定义。"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid


@dataclass
class TraceContext:
    """请求级 TraceContext（最小实现）。"""

    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    stages: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def record_stage(self, name: str, details: Optional[Dict[str, Any]] = None) -> None:
        """记录阶段信息。"""
        payload = {"name": name, "details": details or {}}
        self.stages.append(payload)

    def finish(self) -> Dict[str, Any]:
        """结束并返回 Trace 数据。"""
        return {
            "trace_id": self.trace_id,
            "stages": self.stages,
            "metrics": self.metrics,
            "started_at": self.started_at,
            "finished_at": time.time(),
        }
