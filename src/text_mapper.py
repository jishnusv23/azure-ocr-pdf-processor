"""
Text Position Mapper - Maps search terms to their positions in OCR results
"""
import json
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextMapper:
    """Maps text search terms to their positions in OCR data"""
    
    def __init__(self, ocr_results: Dict[str, Any]):
        """
        Initialize with OCR results
        
        Args:
            ocr_results: Dictionary containing OCR data with text and positions
        """
        self.ocr_results = ocr_results
        self.text_blocks = ocr_results.get('text_blocks', [])
        self.all_text = ocr_results.get('all_text', '')
        
        logger.info(f"✅ TextMapper initialized with {len(self.text_blocks)} text blocks")
    
    def search_text(self, search_term: str, case_sensitive: bool = False) -> List[Dict[str, Any]]:
        """
        Search for a text term in OCR results
        
        Args:
            search_term: Text to search for
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            List of dictionaries containing matched text and positions
        """
        matches = []
        
        search_lower = search_term if case_sensitive else search_term.lower()
        
        for block in self.text_blocks:
            text = block['text']
            text_compare = text if case_sensitive else text.lower()
            
            if search_lower in text_compare:
                match = {
                    'text': text,
                    'search_term': search_term,
                    'bounding_box': block['bounding_box'],
                    'confidence': block.get('confidence', None)
                }
                matches.append(match)
        
        logger.info(f"🔍 Found {len(matches)} matches for '{search_term}'")
        return matches
    
    def search_multiple(self, search_terms: List[str], case_sensitive: bool = False) -> Dict[str, List[Dict[str, Any]]]:
        """
        Search for multiple terms
        
        Args:
            search_terms: List of terms to search
            case_sensitive: Whether to perform case-sensitive search
            
        Returns:
            Dictionary mapping each search term to its matches
        """
        results = {}
        
        for term in search_terms:
            results[term] = self.search_text(term, case_sensitive)
        
        total_matches = sum(len(matches) for matches in results.values())
        logger.info(f"🔍 Searched {len(search_terms)} terms, found {total_matches} total matches")
        
        return results
    
    def get_text_at_position(self, x: float, y: float, tolerance: float = 10) -> Optional[Dict[str, Any]]:
        """
        Get text at a specific position
        
        Args:
            x: X coordinate
            y: Y coordinate
            tolerance: Tolerance in pixels
            
        Returns:
            Text block at that position or None
        """
        for block in self.text_blocks:
            bbox = block['bounding_box']
            
            if (bbox['left'] - tolerance <= x <= bbox['right'] + tolerance and
                bbox['top'] - tolerance <= y <= bbox['bottom'] + tolerance):
                return block
        
        return None
    
    def detect_columns(self, x_tolerance: float = 20) -> List[List[Dict[str, Any]]]:
        """
        Detect columns in the document based on X positions
        
        Args:
            x_tolerance: Tolerance for grouping text blocks into columns
            
        Returns:
            List of columns, each containing text blocks
        """
        if not self.text_blocks:
            return []
        
        # Sort by X position
        sorted_blocks = sorted(self.text_blocks, key=lambda b: b['bounding_box']['left'])
        
        columns = []
        current_column = [sorted_blocks[0]]
        current_x = sorted_blocks[0]['bounding_box']['left']
        
        for block in sorted_blocks[1:]:
            block_x = block['bounding_box']['left']
            
            if abs(block_x - current_x) <= x_tolerance:
                # Same column
                current_column.append(block)
            else:
                # New column
                columns.append(current_column)
                current_column = [block]
                current_x = block_x
        
        # Add last column
        if current_column:
            columns.append(current_column)
        
        logger.info(f"📊 Detected {len(columns)} columns")
        return columns
    
    def detect_tables(self, y_gap_threshold: float = 30) -> List[Dict[str, Any]]:
        """
        Detect tables based on Y position gaps
        
        Args:
            y_gap_threshold: Minimum gap to separate tables
            
        Returns:
            List of detected tables with their boundaries
        """
        if not self.text_blocks:
            return []
        
        # Sort by Y position (top to bottom)
        sorted_blocks = sorted(self.text_blocks, key=lambda b: b['bounding_box']['top'])
        
        tables = []
        current_table_blocks = [sorted_blocks[0]]
        
        for i in range(1, len(sorted_blocks)):
            prev_block = sorted_blocks[i - 1]
            curr_block = sorted_blocks[i]
            
            prev_bottom = prev_block['bounding_box']['bottom']
            curr_top = curr_block['bounding_box']['top']
            
            gap = curr_top - prev_bottom
            
            if gap > y_gap_threshold:
                # End current table
                table = self._create_table_structure(current_table_blocks)
                tables.append(table)
                current_table_blocks = [curr_block]
            else:
                current_table_blocks.append(curr_block)
        
        # Add last table
        if current_table_blocks:
            table = self._create_table_structure(current_table_blocks)
            tables.append(table)
        
        logger.info(f"📊 Detected {len(tables)} tables")
        return tables
    
    def _create_table_structure(self, blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create table structure from blocks"""
        if not blocks:
            return {}
        
        # Calculate table boundaries
        min_x = min(b['bounding_box']['left'] for b in blocks)
        max_x = max(b['bounding_box']['right'] for b in blocks)
        min_y = min(b['bounding_box']['top'] for b in blocks)
        max_y = max(b['bounding_box']['bottom'] for b in blocks)
        
        return {
            'bounding_box': {
                'left': min_x,
                'top': min_y,
                'right': max_x,
                'bottom': max_y,
                'width': max_x - min_x,
                'height': max_y - min_y
            },
            'blocks': blocks,
            'block_count': len(blocks)
        }
    
    def highlight_search_results(self, search_results: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """
        Create highlight data for frontend visualization
        
        Args:
            search_results: Results from search_multiple()
            
        Returns:
            Dictionary with highlight positions for each search term
        """
        highlights = {
            'search_terms': [],
            'highlights': []
        }
        
        for term, matches in search_results.items():
            highlights['search_terms'].append(term)
            
            for match in matches:
                highlight = {
                    'text': match['text'],
                    'search_term': term,
                    'position': match['bounding_box'],
                    'confidence': match.get('confidence')
                }
                highlights['highlights'].append(highlight)
        
        return highlights


if __name__ == "__main__":
    # Test the text mapper
    from config.config import OCR_RESULTS_DIR
    
    # Find first OCR result file
    ocr_files = list(OCR_RESULTS_DIR.glob("*.json"))
    
    if ocr_files:
        test_file = ocr_files[0]
        logger.info(f"\n🧪 Testing with: {test_file.name}")
        
        # Load OCR results
        with open(test_file, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
        
        # Initialize mapper
        mapper = TextMapper(ocr_results)
        
        # Test search
        test_terms = ['862909', 'B3219', 'MDG1234']
        results = mapper.search_multiple(test_terms)
        
        print(f"\n📊 Search Results:")
        for term, matches in results.items():
            print(f"   '{term}': {len(matches)} matches")
            for match in matches[:2]:  # Show first 2 matches
                bbox = match['bounding_box']
                print(f"      - Position: ({bbox['left']:.0f}, {bbox['top']:.0f})")
        
        # Detect columns
        columns = mapper.detect_columns()
        print(f"\n📊 Detected {len(columns)} columns")
        
        # Detect tables
        tables = mapper.detect_tables()
        print(f"📊 Detected {len(tables)} tables")
    else:
        logger.warning(f"⚠️  No OCR result files found in: {OCR_RESULTS_DIR}")