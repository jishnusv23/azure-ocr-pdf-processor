"""
PDF Highlighter Module - FIXED VERSION
Properly handles both component and flat data structures
"""
import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PDFHighlighter:
    """Highlights bounding boxes in PDF files"""
    
    def __init__(self, opacity: float = 0.4):
        self.opacity = opacity
        self.highlight_color = (1.0, 1.0, 0.0)  # Yellow
    
    def highlight_with_rectangles(self, pdf_path: str, extraction_result: Dict[str, Any],
                                   output_path: str = None, ocr_results: Dict[str, Any] = None) -> str:
        """
        Draw colored rectangles with borders
        FIXED: Properly handles BOTH component data and flat data with bounding boxes
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not ocr_results:
            raise ValueError("ocr_results parameter is required!")
        
        # Handle both key names
        img_dims = ocr_results.get('image_size') or ocr_results.get('image_dimensions', {})
        img_width = img_dims.get('width')
        img_height = img_dims.get('height')
        
        if not img_width or not img_height:
            raise ValueError("OCR results missing image_size or image_dimensions!")
        
        if not output_path:
            output_path = pdf_path.parent / f"{pdf_path.stem}_highlighted_boxes.pdf"
        else:
            output_path = Path(output_path)
        
        logger.info(f"📄 Opening PDF: {pdf_path.name}")
        
        try:
            doc = fitz.open(str(pdf_path))
            page = doc[0]
            
            page_width = page.rect.width
            page_height = page.rect.height
            
            logger.info(f"   PDF Page size: {page_width:.0f} x {page_height:.0f}")
            logger.info(f"   OCR Image size: {img_width:.0f} x {img_height:.0f}")
            
            scale_x = page_width / img_width
            scale_y = page_height / img_height
            
            logger.info(f"   Scale factors: X={scale_x:.6f}, Y={scale_y:.6f}\n")
            
            # Get fields
            fields = extraction_result.get('fields', extraction_result)
            highlight_count = 0
            
            # Component types
            component_types = ['Airframe', 'Engine1', 'Engine2', 'APU',
                             'LandingGearLeft', 'LandingGearRight', 'LandingGearNose']
            
            # Check if we have component structure
            has_components = any(comp in fields for comp in component_types)
            
            if has_components:
                logger.info("🔧 Component data detected, highlighting component fields...")
                
                for comp_type in component_types:
                    comp_data = fields.get(comp_type)
                    if comp_data and isinstance(comp_data, dict):
                        logger.info(f"\n   📦 {comp_type}:")
                        
                        for field_name, field_data in comp_data.items():
                            bbox = self._extract_bbox(field_data)
                            
                            if bbox:
                                full_field_name = f"{comp_type}.{field_name}"
                                self._draw_rectangle(page, bbox, full_field_name, scale_x, scale_y)
                                highlight_count += 1
            else:
                logger.info("📋 Flat data structure detected, highlighting top-level fields...")
                
                for field_name, field_data in fields.items():
                    bbox = self._extract_bbox(field_data)
                    
                    if bbox:
                        self._draw_rectangle(page, bbox, field_name, scale_x, scale_y)
                        highlight_count += 1
                    else:
                        logger.warning(f"      ⚠️  {field_name}: No bounding box found")
            
            doc.save(str(output_path))
            doc.close()
            
            logger.info(f"\n✅ Drew {highlight_count} yellow rectangles")
            logger.info(f"💾 Saved to: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error drawing rectangles: {str(e)}")
            raise
    
    def _extract_bbox(self, field_data: Any) -> Dict[str, float]:
        """
        Extract bounding box from field data
        Handles multiple structures:
        1. {'value': X, 'bounding_box': {...}}  <- Component structure
        2. {'bounding_box': {...}}              <- Direct bbox
        3. Direct bbox dict with keys: top, left, width, height
        """
        if field_data is None:
            return None
        
        # Case 1: Nested structure with 'bounding_box' key
        if isinstance(field_data, dict):
            if 'bounding_box' in field_data:
                return field_data['bounding_box']
            
            # Case 2: Direct bbox dict (has required keys)
            required_keys = {'top', 'left', 'width', 'height'}
            if required_keys.issubset(field_data.keys()):
                return field_data
        
        return None
    
    def _draw_rectangle(self, page, bbox: Dict[str, float], field_name: str,
                        scale_x: float = 1.0, scale_y: float = 1.0):
        """Draw a yellow rectangle on the page"""
        
        orig_left = bbox.get('left', 0)
        orig_top = bbox.get('top', 0)
        orig_width = bbox.get('width', 0)
        orig_height = bbox.get('height', 0)
        
        left = orig_left * scale_x
        top = orig_top * scale_y
        width = orig_width * scale_x
        height = orig_height * scale_y
        
        right = left + width
        bottom = top + height
        
        logger.info(f"      {field_name}:")
        logger.info(f"         OCR: ({orig_left:.0f}, {orig_top:.0f}) {orig_width:.0f}x{orig_height:.0f}")
        logger.info(f"         PDF: ({left:.1f}, {top:.1f}) -> ({right:.1f}, {bottom:.1f})")
        
        rect = fitz.Rect(left, top, right, bottom)
        
        page.draw_rect(
            rect, 
            color=self.highlight_color,
            fill=self.highlight_color,
            fill_opacity=0.3,
            width=2
        )


def highlight_extraction_in_pdf(pdf_path: str, extraction_result: Dict[str, Any],
                                output_path: str = None, method: str = "rectangle",
                                ocr_results: Dict[str, Any] = None) -> str:
    """Convenience function to highlight extraction results in PDF"""
    if not ocr_results:
        raise ValueError("ocr_results parameter is required for proper coordinate scaling!")
    
    highlighter = PDFHighlighter(opacity=0.4)
    return highlighter.highlight_with_rectangles(pdf_path, extraction_result, output_path, ocr_results)