from datetime import datetime
from gymnasium import spaces
from pydantic import BaseModel, Field

class SimpleEnv(BaseModel):

    action_space: spaces = Field(description = "action space with gym syntax")
    observation_space: spaces
    trading_days: list[tuple[datetime, datetime]] = Field(description = "list of start and end datetime")
    commission_cfg: dict
    redis_host: str
    redis_port: int
    tick_decimal: float
    ticker: str
    suffix_ticker: str
    max_b_size: int =  Field(description = "max buy positions")
    max_s_size: int = Field(description = "max sell positions")