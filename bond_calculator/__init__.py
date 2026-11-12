"""Помесячная экономика четырёх каналов размещения облигаций."""

from .models import Deal, SystemParameters, RatePeriod, load_deals, load_system
from .channels import (
    ChannelResult, MonthlyResult, AllocationResult,
    institutional_placement, retail_placement, balance_for_resale,
    hold_to_maturity, calculate_allocation,
)
from .returns import IRRResult, ReturnMetrics, calculate_irr, calculate_returns

__all__ = [
    "Deal", "SystemParameters", "RatePeriod", "load_deals", "load_system",
    "ChannelResult", "MonthlyResult", "AllocationResult",
    "institutional_placement", "retail_placement", "balance_for_resale",
    "hold_to_maturity", "calculate_allocation",
    "IRRResult", "ReturnMetrics", "calculate_irr", "calculate_returns",
]
