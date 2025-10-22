"""
Simple script to process a single PDF file
This is a simplified version for quick testing
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_processor import PDFProcessor
from src.azure_ocr import AzureOCR
from config.config import OCR_RESULTS_DIR, VISUALIZATIONS_DIR


def main():
    """Process a single PDF file"""
    
    if len(sys.argv) < 2:
        print("Usage: python scripts/process_single_pdf.py <path_to_pdf>")
        print("\nExample:")
        print("  python scripts/process_single_pdf.py data/sample.pdf")
        sys.exit(1)
    
    pdf_path = Path(sys.argv[1])
    
    if not pdf_path.exists():
        print(f"❌ Error: PDF file not found: {pdf_path}")
        sys.exit(1)
    
    print(f"\n🚀 Processing: {pdf_path.name}")
    print("="*50)
    
    # Step 1: Convert PDF to images
    print("\n📄 Converting PDF to images...")
    processor = PDFProcessor()
    images = processor.convert_pdf_to_images(str(pdf_path))
    print(f"✅ Converted {len(images)} pages")
    
    # Step 2: Process with OCR
    print("\n🔍 Processing with Azure OCR...")
    ocr = AzureOCR()
    
    for page_num, image in images:
        print(f"\n   Processing page {page_num}...")
        
        # Run OCR
        results = ocr.process_image(image)
        
        # Save results
        output_filename = f"{pdf_path.stem}_page_{page_num}_ocr.json"
        output_path = OCR_RESULTS_DIR / output_filename
        ocr.save_results(results, str(output_path))
        
        # Save image
        vis_filename = f"{pdf_path.stem}_page_{page_num}.png"
        vis_path = VISUALIZATIONS_DIR / vis_filename
        processor.save_image(image, str(vis_path))
        
        print(f"   ✅ Page {page_num}: {len(results['text_blocks'])} text blocks")
        print(f"      JSON: {output_path}")
        print(f"      Image: {vis_path}")
    
    print("\n" + "="*50)
    print("🎉 Processing completed!")
    print(f"   Results: {OCR_RESULTS_DIR}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()