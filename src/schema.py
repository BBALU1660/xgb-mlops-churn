from pydantic import BaseModel
from typing import List, Dict, Any

class Row(BaseModel):
    data: Dict[str, Any]

class PredictRequest(BaseModel):
    rows: List[Row]

class PredictResponse(BaseModel):
    probs: List[float]
