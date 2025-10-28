"""
Text Mapper - Maps OCR results to simplified text blocks
"""
import logging
from typing import List, Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextMapper:
    """Maps Azure OCR results to simplified text blocks"""
    
    def __init__(self):
        """Initialize Text Mapper"""
        logger.info("✅ Text Mapper initialized")
    
    def map_text_blocks(self, ocr_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Convert OCR results to simplified text blocks
        
        Args:
            ocr_results: OCR results from Azure Computer Vision
            
        Returns:
            List of simplified text blocks with text and bounding boxes
        """
        text_blocks = ocr_results.get('text_blocks', [])
        
        if not text_blocks:
            logger.warning("⚠️  No text blocks found in OCR results")
            return []
        
        logger.info(f"📋 Mapping {len(text_blocks)} text blocks")
        
        # Text blocks are already in the correct format from Azure OCR
        return text_blocks
    
    def filter_blocks_by_area(self, text_blocks: List[Dict[str, Any]], 
                             min_x: float, min_y: float, 
                             max_x: float, max_y: float) -> List[Dict[str, Any]]:
        """
        Filter text blocks by bounding area
        
        Args:
            text_blocks: List of text blocks
            min_x, min_y: Top-left corner of filter area
            max_x, max_y: Bottom-right corner of filter area
            
        Returns:
            Filtered list of text blocks
        """
        filtered = []
        
        for block in text_blocks:
            bbox = block.get('bounding_box', {})
            block_left = bbox.get('left', 0)
            block_top = bbox.get('top', 0)
            block_right = bbox.get('right', 0)
            block_bottom = bbox.get('bottom', 0)
            
            # Check if block overlaps with filter area
            if (block_left >= min_x and block_right <= max_x and
                block_top >= min_y and block_bottom <= max_y):
                filtered.append(block)
        
        logger.info(f"🔍 Filtered {len(filtered)} blocks in area "
                   f"({min_x:.0f},{min_y:.0f}) to ({max_x:.0f},{max_y:.0f})")
        
        return filtered
    
    def search_text(self, text_blocks: List[Dict[str, Any]], 
                   search_term: str, 
                   case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for text in blocks
        
        Args:
            text_blocks: List of text blocks
            search_term: Text to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            List of blocks containing the search term
        """
        matches = []
        
        for block in text_blocks:
            text = block.get('text', '')
            
            if case_sensitive:
                if search_term in text:
                    matches.append(block)
            else:
                if search_term.lower() in text.lower():
                    matches.append(block)
        
        logger.info(f"🔍 Found {len(matches)} blocks containing '{search_term}'")
        
        return matches
    
    def group_blocks_by_line(self, text_blocks: List[Dict[str, Any]], 
                           y_tolerance: float = 5.0) -> List[List[Dict[str, Any]]]:
        """
        Group text blocks into lines based on vertical position
        
        Args:
            text_blocks: List of text blocks
            y_tolerance: Maximum vertical distance to consider blocks on same line
            
        Returns:
            List of lines, where each line is a list of blocks
        """
        if not text_blocks:
            return []
        
        # Sort blocks by vertical position (top)
        sorted_blocks = sorted(text_blocks, 
                              key=lambda b: b.get('bounding_box', {}).get('top', 0))
        
        lines = []
        current_line = [sorted_blocks[0]]
        current_y = sorted_blocks[0].get('bounding_box', {}).get('top', 0)
        
        for block in sorted_blocks[1:]:
            block_y = block.get('bounding_box', {}).get('top', 0)
            
            if abs(block_y - current_y) <= y_tolerance:
                # Same line
                current_line.append(block)
            else:
                # New line
                # Sort current line by horizontal position
                current_line.sort(key=lambda b: b.get('bounding_box', {}).get('left', 0))
                lines.append(current_line)
                
                current_line = [block]
                current_y = block_y
        
        # Add last line
        if current_line:
            current_line.sort(key=lambda b: b.get('bounding_box', {}).get('left', 0))
            lines.append(current_line)
        
        logger.info(f"📏 Grouped {len(text_blocks)} blocks into {len(lines)} lines")
        
        return lines
    
    def get_text_by_line(self, text_blocks: List[Dict[str, Any]], 
                        y_tolerance: float = 5.0) -> List[str]:
        """
        Get text grouped by lines
        
        Args:
            text_blocks: List of text blocks
            y_tolerance: Maximum vertical distance to consider blocks on same line
            
        Returns:
            List of strings, one per line
        """
        lines = self.group_blocks_by_line(text_blocks, y_tolerance)
        
        line_texts = []
        for line in lines:
            line_text = ' '.join(block.get('text', '') for block in line)
            line_texts.append(line_text)
        
        return line_texts


if __name__ == "__main__":
    # Test text mapper
    print("✅ Text Mapper module loaded successfully")