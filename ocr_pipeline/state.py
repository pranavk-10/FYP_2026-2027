from pydantic import BaseModel, Field
from typing import List, Optional

class BloodMarker(BaseModel):
    value: float = Field(description="The numerical value of the test result")
    unit: str = Field(description="The unit of measurement (e.g., g/dL, cells/mcL)")
    flag: str = Field(description="Strictly 'high', 'low', or 'normal' based on the reference range")

class CBCReportJSON(BaseModel):
    hemoglobin: Optional[BloodMarker] = Field(None, description="Hemoglobin (Hb) levels")
    wbc_count: Optional[BloodMarker] = Field(None, description="White Blood Cell (WBC) count")
    platelets: Optional[BloodMarker] = Field(None, description="Platelet count")
    other_abnormalities: List[str] = Field(default_factory=list, description="Any other markers flagged as abnormal")