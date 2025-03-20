from pydantic import BaseModel, Field
from typing import List

class FinancialHealth(BaseModel):
    classification: str = Field(..., description="The overall financial health classification (Excellent, Good, Fair, Poor, Critical)")
    reasons: List[str] = Field(..., description="List of reasons supporting the classification")
    recommendations: List[str] = Field(..., description="List of recommendations for improvement")
    