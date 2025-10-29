"""
Pydantic models for extraction service
"""

from pydantic import BaseModel, Field, computed_field
from typing import Optional, List, Dict, Any
from enum import Enum


class DocumentType(str, Enum):
    STANDALONE_ASSETS = "standalone_assets"
    FLIGHT_INFO = "flight_info"
    COMPONENT_DATA = "component_data"


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

    SerialNumber_Original: Optional[str] = Field(default=None)
    SerialNumber_Original_bbox: Optional[BoundingBox] = Field(default=None)
    
    location: Optional[str] = Field(default=None)
    location_bbox: Optional[BoundingBox] = Field(default=None)
    
    derate: Optional[float] = Field(default=None)
    
    extraction_confidence: Optional[float] = Field(default=None, ge=0, le=1)
    
    # Helper methods make it easy to use
    def to_field_dict(self) -> Dict[str, Dict[str, Any]]:
        """Convert to field-oriented structure"""
        fields = {}
        for field_name in ['TSN', 'CSN', 'MonthlyUtil_Hrs', 'MonthlyUtil_Cyc', 
                           'SerialNumber', 'SerialNumber_Original', 'location', 'derate']:
            value = getattr(self, field_name, None)
            bbox = getattr(self, f"{field_name}_bbox", None)
            
            if value is not None:
                fields[field_name] = {
                    'value': value.value if isinstance(value, AttachmentStatus) else value,
                    'bounding_box': bbox.dict() if bbox else None
                }
        
        return fields


class ExtractedComponentData(BaseModel):
    """Extracted component data with utilization metrics - MULTI-PAGE SUPPORT"""
    
    Airframe: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Airframe component with utilization metrics, TSN/CSN values. Extract SerialNumber (MSN), TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc with bounding boxes including page_number."
    )
    Engine1: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Engine 1 component (Position 1, 1000EM1). Extract SerialNumber, SerialNumber_Original, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, location with bounding boxes including page_number."
    )
    Engine2: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Engine 2 component (Position 2, 1000EM2). Extract SerialNumber, SerialNumber_Original, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, location with bounding boxes including page_number."
    )
    APU: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Auxiliary Power Unit component. Extract SerialNumber, SerialNumber_Original, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc, location with bounding boxes including page_number."
    )
    LandingGearLeft: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Left Landing Gear component (Main Gear 1). Extract SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc with bounding boxes including page_number."
    )
    LandingGearRight: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Right Landing Gear component (Main Gear 2). Extract SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc with bounding boxes including page_number."
    )
    LandingGearNose: Optional[ComponentData] = Field(
        default_factory=ComponentData,
        description="Nose Landing Gear component. Extract SerialNumber, TSN, CSN, MonthlyUtil_Hrs, MonthlyUtil_Cyc with bounding boxes including page_number."
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