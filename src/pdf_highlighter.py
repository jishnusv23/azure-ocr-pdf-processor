"""
PDF Highlighter Module - Debug Version
Highlights extracted fields in PDF based on bounding box coordinates
"""
import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Dict, Any

logger = logging.getLogger(__name__)


class PDFHighlighter:
    """Highlights bounding boxes in PDF files"""
    
    def __init__(self, opacity: float = 0.4):
        """
        Initialize PDF Highlighter
        
        Args:
            opacity: Highlight opacity (0.0 to 1.0)
        """
        self.opacity = opacity
        self.highlight_color = (1.0, 1.0, 0.0)  # Yellow
    
    def highlight_pdf(self, pdf_path: str, extraction_result: Dict[str, Any], 
                      output_path: str = None, ocr_results: Dict[str, Any] = None) -> str:
        """
        Highlight extracted fields in PDF with proper coordinate scaling
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not ocr_results:
            raise ValueError("ocr_results parameter is required for proper coordinate scaling!")
        
        if not output_path:
            output_path = pdf_path.parent / f"{pdf_path.stem}_highlighted.pdf"
        else:
            output_path = Path(output_path)
        
        logger.info(f"📄 Opening PDF: {pdf_path.name}")
        
        try:
            # Open PDF
            doc = fitz.open(str(pdf_path))
            page = doc[0]
            
            page_width = page.rect.width
            page_height = page.rect.height
            logger.info(f"   PDF Page size: {page_width:.0f} x {page_height:.0f}")
            
            # Get image dimensions from OCR results
            img_dims = ocr_results.get('image_size', {})
            img_width = img_dims.get('width')
            img_height = img_dims.get('height')
            
            if not img_width or not img_height:
                raise ValueError("OCR results missing image_size (width/height)!")
            
            logger.info(f"   OCR Image size: {img_width:.0f} x {img_height:.0f}")
            
            # Calculate scaling factors
            scale_x = page_width / img_width
            scale_y = page_height / img_height
            
            logger.info(f"   Scale factors: X={scale_x:.6f}, Y={scale_y:.6f}")
            logger.info("")
            
            # DIAGNOSTIC: Show what the coordinates look like
            logger.info("="*70)
            logger.info("🔍 DIAGNOSTIC INFO:")
            logger.info("="*70)
            
            fields = extraction_result.get('fields', {})
            
            # Show first field as example
            if fields:
                first_field = list(fields.keys())[0]
                first_bbox = fields[first_field].get('bounding_box', {})
                
                orig_left = first_bbox.get('left', 0)
                orig_top = first_bbox.get('top', 0)
                orig_width = first_bbox.get('width', 0)
                
                scaled_left = orig_left * scale_x
                scaled_top = orig_top * scale_y
                scaled_width = orig_width * scale_x
                
                logger.info(f"Sample field: {first_field}")
                logger.info(f"  OCR coordinates: left={orig_left}, top={orig_top}, width={orig_width}")
                logger.info(f"  Scaled to PDF: left={scaled_left:.1f}, top={scaled_top:.1f}, width={scaled_width:.1f}")
                logger.info(f"  Page width: {page_width}, so {scaled_left:.1f} is {(scaled_left/page_width)*100:.1f}% across")
                logger.info("")
                
                # Check if coordinates seem reasonable
                if scaled_left > page_width or scaled_top > page_height:
                    logger.warning("⚠️  WARNING: Scaled coordinates are outside the PDF page!")
                    logger.warning(f"   This means the scaling factors are wrong.")
                    logger.warning(f"   The OCR image dimensions might be incorrect.")
                    logger.info("")
            
            logger.info("="*70)
            logger.info("")
            
            highlight_count = 0
            
            # Highlight flat fields
            for field_name, field_data in fields.items():
                if field_name == 'engines':
                    continue
                
                bbox = field_data.get('bounding_box')
                if bbox:
                    self._add_highlight(page, bbox, field_name, scale_x, scale_y)
                    highlight_count += 1
            
            # Highlight engine fields
            if 'engines' in fields:
                for engine_idx, engine in enumerate(fields['engines'], 1):
                    for field_name, field_data in engine.items():
                        bbox = field_data.get('bounding_box')
                        if bbox:
                            self._add_highlight(page, bbox, f"engine_{field_name}", scale_x, scale_y)
                            highlight_count += 1
            
            # Save highlighted PDF
            doc.save(str(output_path))
            doc.close()
            
            logger.info("")
            logger.info(f"✅ Created {highlight_count} highlights")
            logger.info(f"💾 Saved to: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error highlighting PDF: {str(e)}")
            raise
    
    def _add_highlight(self, page, bbox: Dict[str, float], field_name: str, 
                       scale_x: float = 1.0, scale_y: float = 1.0):
        """Add a yellow highlight to the page with proper scaling"""
        
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
        
        logger.info(f"   {field_name}:")
        logger.info(f"      OCR: ({orig_left:.0f}, {orig_top:.0f}) {orig_width:.0f}x{orig_height:.0f}")
        logger.info(f"      PDF: ({left:.1f}, {top:.1f}) {width:.1f}x{height:.1f}")
        logger.info(f"      Rect: [{left:.1f}, {top:.1f}, {right:.1f}, {bottom:.1f}]")
        
        rect = fitz.Rect(left, top, right, bottom)
        
        highlight = page.add_highlight_annot(rect)
        highlight.set_colors(stroke=self.highlight_color)
        highlight.set_opacity(self.opacity)
        highlight.update()
    
    def highlight_with_rectangles(self, pdf_path: str, extraction_result: Dict[str, Any],
                                   output_path: str = None, ocr_results: Dict[str, Any] = None) -> str:
        """
        Draw colored rectangles with borders (more visible than highlights)
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        if not ocr_results:
            raise ValueError("ocr_results parameter is required!")
        
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
            
            img_dims = ocr_results.get('image_size', {})
            img_width = img_dims.get('width')
            img_height = img_dims.get('height')
            
            if not img_width or not img_height:
                raise ValueError("OCR results missing image_size!")
            
            logger.info(f"   OCR Image size: {img_width:.0f} x {img_height:.0f}")
            
            scale_x = page_width / img_width
            scale_y = page_height / img_height
            
            logger.info(f"   Scale factors: X={scale_x:.6f}, Y={scale_y:.6f}")
            logger.info("")
            
            fields = extraction_result.get('fields', {})
            highlight_count = 0
            
            for field_name, field_data in fields.items():
                if field_name == 'engines':
                    continue
                
                bbox = field_data.get('bounding_box')
                if bbox:
                    self._draw_rectangle(page, bbox, field_name, scale_x, scale_y)
                    highlight_count += 1
            
            if 'engines' in fields:
                for engine_idx, engine in enumerate(fields['engines'], 1):
                    for field_name, field_data in engine.items():
                        bbox = field_data.get('bounding_box')
                        if bbox:
                            self._draw_rectangle(page, bbox, f"engine_{field_name}", scale_x, scale_y)
                            highlight_count += 1
            
            doc.save(str(output_path))
            doc.close()
            
            logger.info("")
            logger.info(f"✅ Drew {highlight_count} yellow rectangles")
            logger.info(f"💾 Saved to: {output_path}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error drawing rectangles: {str(e)}")
            raise
    
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
        
        logger.info(f"   {field_name}:")
        logger.info(f"      OCR: ({orig_left:.0f}, {orig_top:.0f}) {orig_width:.0f}x{orig_height:.0f}")
        logger.info(f"      PDF: ({left:.1f}, {top:.1f}) {width:.1f}x{height:.1f}")
        
        rect = fitz.Rect(left, top, right, bottom)
        
        page.draw_rect(
            rect, 
            color=self.highlight_color,
            fill=self.highlight_color,
            fill_opacity=0.3,
            width=2
        )


def highlight_extraction_in_pdf(pdf_path: str, extraction_result: Dict[str, Any],
                                output_path: str = None, method: str = "highlight",
                                ocr_results: Dict[str, Any] = None) -> str:
    """
    Convenience function to highlight extraction results in PDF
    """
    if not ocr_results:
        raise ValueError("ocr_results parameter is required for proper coordinate scaling!")
    
    highlighter = PDFHighlighter(opacity=0.4)
    
    if method == "rectangle":
        return highlighter.highlight_with_rectangles(pdf_path, extraction_result, output_path, ocr_results)
    else:
        return highlighter.highlight_pdf(pdf_path, extraction_result, output_path, ocr_results)