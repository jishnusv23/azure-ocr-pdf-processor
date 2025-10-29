"""
PDF Highlighter - COMPLETE MULTI-PAGE SUPPORT
Highlights fields across ALL pages in ONE output PDF
Location: src/pdf_highlighter.py
"""
import fitz  # PyMuPDF
import logging
from pathlib import Path
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class PDFHighlighter:
    """Highlights bounding boxes across multiple pages in PDF files"""
    
    def __init__(self, opacity: float = 0.4):
        self.opacity = opacity
        self.highlight_color = (1.0, 1.0, 0.0)  # Yellow
    
    def highlight_multipage(self, pdf_path: str, fields_by_page: Dict[int, List[Dict]], 
                           output_path: str, all_ocr_data: List[Dict[str, Any]]) -> str:
        """
        Highlight fields across MULTIPLE pages in ONE output PDF
        
        Args:
            pdf_path: Path to input PDF
            fields_by_page: Dict mapping page_number -> list of fields with bounding boxes
                           {1: [{'field': 'TSN', 'value': '16300', 'bounding_box': {...}}], 
                            2: [{'field': 'CSN', 'value': '8200', 'bounding_box': {...}}]}
            output_path: Path for output PDF
            all_ocr_data: List of OCR data dicts (one per page) with image dimensions
        
        Returns:
            Path to highlighted PDF
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        
        output_path = Path(output_path)
        
        logger.info(f"📄 Opening PDF: {pdf_path.name}")
        logger.info(f"🎯 Highlighting across pages: {sorted(fields_by_page.keys())}")
        
        try:
            doc = fitz.open(str(pdf_path))
            
            # Create mapping of page_number -> OCR dimensions
            ocr_dims_by_page = {}
            for ocr_data in all_ocr_data:
                page_num = ocr_data['page_number']
                ocr_dims_by_page[page_num] = ocr_data['image_dimensions']
            
            total_highlights = 0
            
            # Process each page that has fields
            for page_num, fields in sorted(fields_by_page.items()):
                logger.info(f"\n{'='*60}")
                logger.info(f"📄 Processing Page {page_num}")
                logger.info(f"{'='*60}")
                
                # Get page object (convert 1-indexed to 0-indexed)
                page_idx = page_num - 1
                
                if page_idx < 0 or page_idx >= len(doc):
                    logger.warning(f"⚠️  Invalid page number {page_num}, skipping...")
                    continue
                
                page = doc[page_idx]
                
                # Get OCR dimensions for this page
                if page_num not in ocr_dims_by_page:
                    logger.warning(f"⚠️  No OCR dimensions for page {page_num}, skipping...")
                    continue
                
                img_dims = ocr_dims_by_page[page_num]
                img_width = img_dims.get('width')
                img_height = img_dims.get('height')
                
                if not img_width or not img_height:
                    logger.warning(f"⚠️  Invalid dimensions for page {page_num}, skipping...")
                    continue
                
                # Calculate scale factors
                page_width = page.rect.width
                page_height = page.rect.height
                
                scale_x = page_width / img_width
                scale_y = page_height / img_height
                
                logger.info(f"   PDF size: {page_width:.0f} x {page_height:.0f}")
                logger.info(f"   OCR size: {img_width:.0f} x {img_height:.0f}")
                logger.info(f"   Scale: X={scale_x:.4f}, Y={scale_y:.4f}")
                logger.info(f"   Fields to highlight: {len(fields)}\n")
                
                # Highlight each field on this page
                for field_data in fields:
                    field_name = field_data.get('field', 'unknown')
                    field_value = field_data.get('value', '')
                    bbox = field_data.get('bounding_box')
                    
                    if not bbox:
                        logger.warning(f"      ⚠️  {field_name}: No bounding box")
                        continue
                    
                    self._draw_rectangle(
                        page, bbox, 
                        f"{field_name}={field_value}", 
                        scale_x, scale_y
                    )
                    total_highlights += 1
            
            # Save the highlighted PDF
            doc.save(str(output_path))
            doc.close()
            
            logger.info(f"\n{'='*60}")
            logger.info(f"✅ Multi-page highlighting complete!")
            logger.info(f"   Total rectangles drawn: {total_highlights}")
            logger.info(f"   Pages highlighted: {sorted(fields_by_page.keys())}")
            logger.info(f"   Output: {output_path}")
            logger.info(f"{'='*60}")
            
            return str(output_path)
            
        except Exception as e:
            logger.error(f"❌ Error in multi-page highlighting: {str(e)}")
            import traceback
            traceback.print_exc()
            raise
    
    def _draw_rectangle(self, page, bbox: Dict[str, float], label: str,
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
        
        logger.info(f"      ✏️  {label}")
        logger.info(f"         OCR: ({orig_left:.0f}, {orig_top:.0f}) {orig_width:.0f}x{orig_height:.0f}")
        logger.info(f"         PDF: ({left:.1f}, {top:.1f}) → ({right:.1f}, {bottom:.1f})")
        
        rect = fitz.Rect(left, top, right, bottom)
        
        page.draw_rect(
            rect, 
            color=self.highlight_color,
            fill=self.highlight_color,
            fill_opacity=self.opacity,
            width=2
        )
    
    # ============================================================================
    # LEGACY METHOD (for backward compatibility)
    # ============================================================================
    
    def highlight_with_rectangles(self, pdf_path: str, extraction_result: Dict[str, Any],
                                   output_path: str = None, ocr_results: Dict[str, Any] = None,
                                   page_number: int = 1) -> str:
        """Legacy single-page highlighting - kept for backward compatibility"""
        logger.warning("⚠️ Using legacy single-page highlighting. Use highlight_multipage() for better results.")
        
        # Convert to multi-page format
        fields = extraction_result.get('fields', extraction_result)
        
        fields_list = []
        for key, value in fields.items():
            if isinstance(value, dict):
                # Component structure
                for field_name, field_data in value.items():
                    if isinstance(field_data, dict) and 'bounding_box' in field_data:
                        fields_list.append({
                            'field': f"{key}.{field_name}",
                            'value': field_data.get('value', ''),
                            'bounding_box': field_data['bounding_box']
                        })
            elif isinstance(value, dict) and 'bounding_box' in value:
                # Flat structure
                fields_list.append({
                    'field': key,
                    'value': value.get('value', ''),
                    'bounding_box': value['bounding_box']
                })
        
        fields_by_page = {page_number: fields_list}
        
        all_ocr_data = [{
            'page_number': page_number,
            'image_dimensions': ocr_results.get('image_dimensions') or ocr_results.get('image_size', {})
        }]
        
        if not output_path:
            pdf_path_obj = Path(pdf_path)
            output_path = pdf_path_obj.parent / f"{pdf_path_obj.stem}_highlighted.pdf"
        
        return self.highlight_multipage(pdf_path, fields_by_page, output_path, all_ocr_data)


def highlight_extraction_in_pdf(pdf_path: str, extraction_result: Dict[str, Any],
                                output_path: str = None, method: str = "rectangle",
                                ocr_results: Dict[str, Any] = None,
                                page_number: int = 1) -> str:
    """Legacy convenience function - kept for backward compatibility"""
    if not ocr_results:
        raise ValueError("ocr_results parameter is required!")
    
    highlighter = PDFHighlighter(opacity=0.4)
    return highlighter.highlight_with_rectangles(
        pdf_path, extraction_result, output_path, ocr_results, page_number
    )