"""
Search for terms across ALL pages of processed PDF files
"""
import sys
import json
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.text_mapper import TextMapper
from config.config import OCR_RESULTS_DIR


def search_all_pages(search_term, pdf_name=None):
    """
    Search for a term across all OCR result pages
    
    Args:
        search_term: Term to search for
        pdf_name: Optional PDF name to limit search (e.g., "sample3")
    """
    
    # Find all JSON files
    if pdf_name:
        # Search only files from specific PDF
        json_files = sorted(OCR_RESULTS_DIR.glob(f"{pdf_name}_page_*.json"))
    else:
        # Search all JSON files
        json_files = sorted(OCR_RESULTS_DIR.glob("*_page_*.json"))
    
    if not json_files:
        print(f"❌ No OCR result files found in: {OCR_RESULTS_DIR}")
        if pdf_name:
            print(f"   Looking for: {pdf_name}_page_*.json")
        return
    
    print(f"\n{'='*70}")
    print(f"🔍 Searching for: '{search_term}'")
    print(f"   Searching {len(json_files)} page(s)")
    print(f"{'='*70}\n")
    
    total_matches = 0
    results_by_file = defaultdict(list)
    
    # Search each page
    for json_file in json_files:
        # Extract page number from filename
        parts = json_file.stem.split('_page_')
        if len(parts) == 2:
            pdf_base = parts[0]
            page_num = parts[1].replace('_ocr', '')
        else:
            pdf_base = json_file.stem
            page_num = "?"
        
        # Load OCR results
        with open(json_file, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
        
        # Search
        mapper = TextMapper(ocr_results)
        matches = mapper.search_text(search_term)
        
        if matches:
            total_matches += len(matches)
            results_by_file[json_file.stem] = {
                'page_num': page_num,
                'pdf_base': pdf_base,
                'matches': matches,
                'mapper': mapper
            }
            
            print(f"✅ Found {len(matches)} match(es) in: {json_file.name}")
            for i, match in enumerate(matches, 1):
                bbox = match['bounding_box']
                print(f"   {i}. '{match['text']}' at ({bbox['left']:.0f}, {bbox['top']:.0f})")
                print(f"      Confidence: {match.get('confidence', 'N/A')}")
        else:
            print(f"❌ No matches in: {json_file.name}")
    
    # Show detailed results
    if total_matches > 0:
        print(f"\n{'='*70}")
        print(f"📊 Detailed Results - Total: {total_matches} matches")
        print(f"{'='*70}\n")
        
        for file_stem, data in results_by_file.items():
            print(f"\n{'─'*70}")
            print(f"📄 Page {data['page_num']} ({data['pdf_base']})")
            print(f"{'─'*70}")
            
            mapper = data['mapper']
            matches = data['matches']
            
            # Detect columns
            columns = mapper.detect_columns(x_tolerance=20)
            
            for match_idx, match in enumerate(matches, 1):
                print(f"\nMatch #{match_idx}: '{match['text']}'")
                
                match_bbox = match['bounding_box']
                match_x = match_bbox['left']
                match_y = match_bbox['top']
                
                print(f"Position: X={match_x:.0f}, Y={match_y:.0f}")
                print(f"Size: {match_bbox['width']:.0f} x {match_bbox['height']:.0f}")
                
                # Find which column this match belongs to
                column_idx = None
                for idx, column in enumerate(columns):
                    for block in column:
                        if (block['text'] == match['text'] and 
                            abs(block['bounding_box']['left'] - match_x) < 5):
                            column_idx = idx
                            break
                    if column_idx is not None:
                        break
                
                if column_idx is not None:
                    print(f"\n📍 Column #{column_idx + 1} Data:")
                    
                    # Sort column data by Y position (top to bottom)
                    column_data = sorted(columns[column_idx], 
                                       key=lambda x: x['bounding_box']['top'])
                    
                    print(f"   All text in this column ({len(column_data)} items):")
                    for i, item in enumerate(column_data[:10], 1):  # Show first 10
                        item_text = item['text']
                        item_y = item['bounding_box']['top']
                        
                        # Highlight the matched item
                        if (item_text == match['text'] and 
                            abs(item['bounding_box']['left'] - match_x) < 5):
                            print(f"   {i:2d}. ⭐ {item_text:<30} (Y={item_y:.0f})")
                        else:
                            print(f"   {i:2d}.    {item_text:<30} (Y={item_y:.0f})")
                    
                    if len(column_data) > 10:
                        print(f"   ... and {len(column_data) - 10} more items")
                    
                    # Column bounding box
                    col_left = min(b['bounding_box']['left'] for b in column_data)
                    col_right = max(b['bounding_box']['right'] for b in column_data)
                    col_top = min(b['bounding_box']['top'] for b in column_data)
                    col_bottom = max(b['bounding_box']['bottom'] for b in column_data)
                    
                    print(f"\n   Column Bounding Box for Frontend:")
                    print(f"      Left: {col_left:.0f}")
                    print(f"      Top: {col_top:.0f}")
                    print(f"      Width: {col_right - col_left:.0f}")
                    print(f"      Height: {col_bottom - col_top:.0f}")
    else:
        print(f"\n❌ No matches found for '{search_term}' in any page")
    
    print(f"\n{'='*70}\n")


def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/search_all_pages.py <search_term> [pdf_name]")
      
        sys.exit(1)
    
    search_term = sys.argv[1]
    pdf_name = sys.argv[2] if len(sys.argv) > 2 else None
    
    search_all_pages(search_term, pdf_name)


if __name__ == "__main__":
    main()