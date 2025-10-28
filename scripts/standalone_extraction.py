"""
Standalone PDF extraction script - NO DATABASE
Input: 1 PDF → Output: Multiple highlighted PDFs (one per serial number)
Pipeline: PDF → Azure OCR → LLM Extraction → PDF Highlighter

Location: scripts/standalone_extraction.py

Usage from project root:
    python scripts/standalone_extraction.py --file data/sample.pdf
    python scripts/standalone_extraction.py -f data/sample.pdf -o results/
"""
import argparse
import sys
from pathlib import Path
import logging
from typing import List, Dict, Any

# Add parent directory to path to import from src/
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pdf_processor import PDFProcessor
from src.azure_ocr import AzureOCR
from src.text_mapper import TextMapper
from src.llm_field_extractor import LLMFieldExtractor
from src.pdf_highlighter import highlight_extraction_in_pdf
from config.config import OUTPUT_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pdf_standalone(pdf_path: str, output_dir: str = None) -> None:
    """
    Standalone PDF processing - NO DATABASE
    Input: 1 PDF → Output: Multiple highlighted PDFs (one per identifier)
    
    Args:
        pdf_path: Path to input PDF file
        output_dir: Output directory for highlighted PDFs (default: data/output)
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"❌ File not found: {pdf_path}")
        return
    
    # Set output directory
    if not output_dir:
        output_dir = OUTPUT_DIR
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info(f"🚀 STANDALONE EXTRACTION PIPELINE (No Database)")
    logger.info(f"   Input PDF: {pdf_path.name}")
    logger.info(f"   Output Dir: {output_dir}")
    logger.info("="*80)
    
    try:
        # Initialize processors
        pdf_processor = PDFProcessor()
        azure_ocr = AzureOCR()
        text_mapper = TextMapper()
        llm_extractor = LLMFieldExtractor()
        
        # Convert PDF to images
        logger.info("\n🖼️  Step 1: Converting PDF to images...")
        images = pdf_processor.pdf_to_images(str(pdf_path))
        logger.info(f"   ✅ Total pages: {len(images)}")
        
        # Store all extraction results
        all_extractions = []
        
        # Process each page
        for page_idx, image in enumerate(images, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"📄 Step 2: Processing Page {page_idx}")
            logger.info(f"{'='*80}")
            
            # Perform Azure OCR
            logger.info("   🔍 Performing Azure OCR...")
            ocr_result = azure_ocr.analyze_image(image)
            
            if not ocr_result:
                logger.warning(f"   ⚠️  No OCR results for page {page_idx}")
                continue
            
            # Map text blocks with bounding boxes
            text_blocks = text_mapper.map_text_blocks(ocr_result)
            
            # Build OCR data structure
            ocr_data = {
                'filename': pdf_path.name,
                'page_number': page_idx,
                'text_blocks': text_blocks,
                'total_blocks': len(text_blocks),
                'image_dimensions': {
                    'width': image.width,
                    'height': image.height
                },
                'image_size': {  # For highlighter
                    'width': image.width,
                    'height': image.height
                }
            }
            
            logger.info(f"   ✅ OCR: Extracted {len(text_blocks)} text blocks")
            logger.info(f"   📐 Image size: {image.width} x {image.height}")
            
            # LLM Extraction
            logger.info(f"\n{'='*80}")
            logger.info(f"🧠 Step 3: LLM Extraction (Page {page_idx})")
            logger.info(f"{'='*80}")
            
            try:
                extraction_results = llm_extractor.extract_all_data(ocr_data)
                
                if not extraction_results:
                    logger.warning("   ⚠️  No identifiers found on this page")
                    continue
                
                logger.info(f"   ✅ Found {len(extraction_results)} identifiers")
                
                # Store results with OCR data
                for result in extraction_results:
                    all_extractions.append({
                        'page': page_idx,
                        'identifier': result['identifier'],
                        'identifier_type': result['identifier_type'],
                        'extraction_result': result,
                        'ocr_data': ocr_data
                    })
                    logger.info(f"      • {result['identifier']} ({result['identifier_type']})")
            
            except Exception as e:
                logger.error(f"   ❌ Error during LLM extraction: {str(e)}", exc_info=True)
        
        # Generate highlighted PDFs
        if not all_extractions:
            logger.warning("\n⚠️  No identifiers found in entire PDF")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info(f"🎨 Step 4: Generating Highlighted PDFs")
        logger.info(f"{'='*80}")
        logger.info(f"   Total identifiers found: {len(all_extractions)}")
        logger.info(f"   Creating {len(all_extractions)} highlighted PDFs...\n")
        
        generated_pdfs = []
        
        for idx, extraction in enumerate(all_extractions, 1):
            identifier = extraction['identifier']
            page_num = extraction['page']
            extraction_result = extraction['extraction_result']
            ocr_data = extraction['ocr_data']
            
            # Generate output filename
            safe_identifier = identifier.replace('/', '-').replace('\\', '-')
            output_filename = f"{pdf_path.stem}_page{page_num}_{safe_identifier}_highlighted.pdf"
            output_path = output_dir / output_filename
            
            logger.info(f"{idx}. Highlighting: {identifier} (Page {page_num})")
            logger.info(f"   Output: {output_filename}")
            
            try:
                # Highlight the PDF
                highlighted_pdf = highlight_extraction_in_pdf(
                    pdf_path=str(pdf_path),
                    extraction_result=extraction_result,
                    output_path=str(output_path),
                    method="rectangle",
                    ocr_results=ocr_data
                )
                
                generated_pdfs.append({
                    'identifier': identifier,
                    'page': page_num,
                    'file': highlighted_pdf,
                    'fields': extraction_result.get('total_fields', 0)
                })
                
                logger.info(f"   ✅ Created: {output_filename}")
                logger.info(f"   📊 Fields highlighted: {extraction_result.get('total_fields', 0)}\n")
            
            except Exception as e:
                logger.error(f"   ❌ Error highlighting {identifier}: {str(e)}\n")
        
        # Summary
        logger.info("="*80)
        logger.info("🎉 EXTRACTION COMPLETED!")
        logger.info("="*80)
        logger.info(f"   📄 Input PDF: {pdf_path.name}")
        logger.info(f"   📄 Pages processed: {len(images)}")
        logger.info(f"   🔍 Identifiers found: {len(all_extractions)}")
        logger.info(f"   ✅ PDFs generated: {len(generated_pdfs)}")
        logger.info(f"\n   📁 Output Directory: {output_dir}")
        logger.info(f"\n   📋 Generated Files:")
        
        for i, pdf_info in enumerate(generated_pdfs, 1):
            logger.info(f"      {i}. {Path(pdf_info['file']).name}")
            logger.info(f"         → {pdf_info['identifier']} (Page {pdf_info['page']}, {pdf_info['fields']} fields)")
        
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"\n❌ Error during pipeline: {str(e)}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Standalone PDF extraction - NO DATABASE (Direct: PDF → OCR → LLM → Highlighted PDFs)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process PDF and generate highlighted PDFs for each identifier
  python standalone_extraction.py --file data/sample.pdf
  
  # Specify custom output directory
  python standalone_extraction.py --file data/sample.pdf --output-dir data/my_output
  
  # Process any PDF
  python standalone_extraction.py --file path/to/your/file.pdf

Output:
  - One highlighted PDF per identifier found
  - No database connection needed
  - Direct pipeline: PDF → OCR → LLM → Highlighted PDFs
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        required=True,
        help='Path to PDF file to process'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        help='Output directory for highlighted PDFs (default: data/output)'
    )
    
    args = parser.parse_args()
    
    # Process PDF
    process_pdf_standalone(
        pdf_path=args.file,
        output_dir=args.output_dir
    )


if __name__ == "__main__":
    main()