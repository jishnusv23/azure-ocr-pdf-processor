from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional,Literal


class FieldData(BaseModel):
    """Individual field data with block reference"""
    block_id: int = Field(description="Index of the text block in OCR results")
    text: str = Field(description="The actual text content")
    field_type: str = Field(description="Type of field (e.g., 'apu_sn', 'engine_tsn')")
    row_context: Optional[str] = Field(None, description="Row label/context if applicable")


class EngineData(BaseModel):
    """Engine-specific data"""
    position: Optional[FieldData] = None
    pn: Optional[FieldData] = Field(None, description="Engine Part Number")
    sn: Optional[FieldData] = Field(None, description="Engine Serial Number")
    sn_original: Optional[FieldData] = Field(None, description="S/N of Original Engine's")
    original_location: Optional[FieldData] = Field(None, description="Present Location of Original Engine")
    tsn: Optional[FieldData] = Field(None, description="Time Since New")
    csn: Optional[FieldData] = Field(None, description="Cycles Since New")
    delta_hrs: Optional[FieldData] = Field(None, description="Delta hours")
    delta_cyc: Optional[FieldData] = Field(None, description="Delta cycles")


class BoundingBox(BaseModel):
    """Overall bounding box for the extracted section"""
    min_x: float
    max_x: float
    min_y: float
    max_y: float


class ExtractionResult(BaseModel):
    """Complete extraction result from LLM"""
    identifier_found: bool = Field(description="Whether the identifier was found")
    identifier: Optional[str] = Field(None, description="The found identifier")
    identifier_type: Optional[Literal[
        "aircraft_registration", "engine_sn", "apu_sn", "msn", "other"
    ]] = Field(None, description="Type of identifier")
    layout_type: Optional[Literal["columnar", "row_based"]] = Field(
        None, description="Document layout type"
    )
    extracted_fields: Dict[str, Any] = Field(
        default_factory=dict, 
        description="Dictionary of extracted fields"
    )
    engines: Optional[List[EngineData]] = Field(
        None, description="List of engine data if multiple engines"
    )
    bounding_box: Optional[BoundingBox] = Field(
        None, description="Overall bounding box"
    )