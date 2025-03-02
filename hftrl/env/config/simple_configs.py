from datetime import datetime
from gymnasium.spaces import Dict
from pydantic import BaseModel, Field
from typing import Any

class ConfigSimpleEnv(BaseModel):
    action_space: Any
    observation_space: Any
    trading_days: dict = Field(
        default = None, description = "dictionary for datetime generation")
    tick_unit: float = Field(default = None, description = "Precise the tick unit for the future price")
    tp: int = Field(default = None, description = "Takeprofit in points for the market order")
    sl: int = Field(default = None, description = "Stoploss in points for the market order")
    commission_cfg: dict
    redis_host: str
    redis_port: int
    tick_decimal: float
    ticker: str
    suffix_ticker: str
    max_size: int = Field(default =1, description = "max positions")
    side_lad: int
    enable_render: bool = Field(default = False, description = "Enable the rendering during evaluation")
    path_render: str = Field(description = "The path were the evaluation will be saved")

class ConfigMarketMakingEnv(BaseModel):
    action_space: Any
    observation_space: Any
    trading_days: dict = Field(
        default = None, description = "dictionary for datetime generation")
    tick_unit: float = Field(default = None, description = "Precise the tick unit for the future price")
    commission_cfg: dict
    redis_host: str
    redis_port: int
    decimal_time: float
    ticker: str
    suffix_ticker: str
    max_open_pos: int
    unit_tick: float
    enable_render: bool = Field(default = False, description = "Enable the rendering during evaluation")
    path_render: str = Field(description = "The path were the evaluation will be saved")
    mt5_reader: Any