from pydantic import BaseModel

class TickerInput(BaseModel):
    ticker: str
