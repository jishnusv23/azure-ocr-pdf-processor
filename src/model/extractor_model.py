"""
Pydantic models for extraction service
"""

from pydantic import BaseModel, Field,computed_field
from typing import Optional, List,Dict,Any
from enum import Enum


class DocumentType(str, Enum):
    STANDALONE_ASSETS = "standalone_assets"
    FLIGHT_INFO = "flight_info"


class DocumentClassification(BaseModel):
    document_type: DocumentType = Field(
        description="Type of document: standalone_assets (component-focused data) or flight_info (aircraft flight data)"
    )
    confidence: float = Field(
        description="Confidence level (0.0 to 1.0) in the classification"
    )
    reasoning: str = Field(
        description="Brief explanation of why this classification was chosen"
    )


class AttachmentStatus(str, Enum):
    ATTACHED = "Attached"
    REMOVED = "Removed"

class BoundingBox(BaseModel):
    left: float = Field(ge=0)
    top: float = Field(ge=0)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    page_number: Optional[int] = Field(default=1, ge=1)
    
    @computed_field
    @property
    def right(self) -> float:
        return self.left + self.width
    
    @computed_field
    @property
    def bottom(self) -> float:
        return self.top + self.height

# class ComponentData(BaseModel):
#     TSN: Optional[float] = Field(default=None, description="Component Time Since New (hours)")
#     CSN: Optional[int] = Field(default=None, description="Component Cycles Since New")
#     MonthlyUtil_Hrs: Optional[float] = Field(default=None, description="Component Monthly Utilization in Hours")
#     MonthlyUtil_Cyc: Optional[int] = Field(default=None, description="Component Monthly Utilization in Cycles")
#     SerialNumber: Optional[str] = Field(default=None, description="Component Serial Number")
#     location: Optional[str] = Field(default=None, description="Component location information (e.g., #1, #2, MSN, tail number)")
#     attachment_status: Optional[AttachmentStatus] = Field(default=None, description="Component attachment status - Installed or Removed")
#     derate: Optional[float] = Field(default=None, description="Engine derate percentage value as decimal (e.g., 19.08 for 19.08%)")
#     extraction_confidence: Optional[float] = Field(default=None, description="Overall confidence score (0.0-1.0) for the extracted component data including decimal precision")

class ComponentData(BaseModel):
    # Direct values (clean API)
    TSN: Optional[float] = Field(default=None)
    TSN_bbox: Optional[BoundingBox] = Field(default=None)
    
    CSN: Optional[int] = Field(default=None)
    CSN_bbox: Optional[BoundingBox] = Field(default=None)
    
    MonthlyUtil_Hrs: Optional[float] = Field(default=None)
    MonthlyUtil_Hrs_bbox: Optional[BoundingBox] = Field(default=None)
    
    MonthlyUtil_Cyc: Optional[int] = Field(default=None)
    MonthlyUtil_Cyc_bbox: Optional[BoundingBox] = Field(default=None)
    
    SerialNumber: Optional[str] = Field(default=None)
    SerialNumber_bbox: Optional[BoundingBox] = Field(default=None)
    
    location: Optional[str] = Field(default=None)
    location_bbox: Optional[BoundingBox] = Field(default=None)
    
    attachment_status: Optional[AttachmentStatus] = Field(default=None)
    derate: Optional[float] = Field(default=None)
    
    
    extraction_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    component_bbox: Optional[BoundingBox] = Field(default=None)
    
    # Helper methods make it easy to use
    def to_field_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to field-oriented structure"""
        fields = {}
        for field_name in ['TSN', 'CSN', 'MonthlyUtil_Hrs', 'MonthlyUtil_Cyc', 
                           'SerialNumber', 'location', 'attachment_status', 'derate']:
            value = getattr(self, field_name)
            bbox = getattr(self, f"{field_name}_bbox", None)
            
            if value is not None:
                fields[field_name] = {
                    'value': value.value if isinstance(value, AttachmentStatus) else value,
                    'bounding_box': bbox.dict() if bbox else None
                }
        
        return fields

class ExtractedComponentData(BaseModel):
    """Extracted component data with utilization metrics"""
    Airframe: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Airframe component with utilization metrics, TSN/CSN values, and status information"
    )
    Engine1: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Engine 1 component with serial report, utilization metrics, TSN/CSN values, and status information"
    )
    Engine2: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Engine 2 component with serial report, utilization metrics, TSN/CSN values, and status information"
    )
    APU: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Auxiliary Power Unit component with utilization metrics, TSN/CSN values, and status information"
    )
    LandingGearLeft: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Left Landing Gear component with serial report, utilization metrics, TSN/CSN values, and status information"
    )
    LandingGearRight: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Right Landing Gear component with serial report, utilization metrics, TSN/CSN values, and status information"
    )
    LandingGearNose: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Nose Landing Gear component with serial report, utilization metrics, TSN/CSN values, and status information"
    )


class StandaloneAssetsData(BaseModel):
    Month: Optional[str] = Field(default=None, description="Reporting period of the Util report in format 'January 2025' or 'May 2025'")
    MSN: Optional[str] = Field(default=None, description="Aircraft Manufacturer Serial Number")
    FlightRegistrationNumber: Optional[str] = Field(default=None, description="Aircraft registration/tail number (if available in the document)")
    ComponentSerialNumber: str = Field(description="Engine or APU component serial number")


class FlightInfo(BaseModel):
    Month: Optional[str] = Field(default=None, description="Reporting period of the Util report in format 'January 2025' or 'May 2025'")
    MSN: Optional[str] = Field(default=None, description="Manufacturer Serial Number")
    AirCraftType: Optional[str] = Field(default=None, description="Aircraft Type")
    RegistrationNumber: Optional[str] = Field(default=None, description="Registration Number of the aircraft")


class StandaloneData(BaseModel):
    StandaloneAssets: List[StandaloneAssetsData]


class FlightData(BaseModel):
    Planes: List[FlightInfo]
