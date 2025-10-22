"""
Main script for processing PDFs with Azure OCR and LLM extraction
"""
import argparse
import sys
import json
from pathlib import Path
import logging
from typing import Optional

from src.pdf_processor import PDFProcessor
from src.azure_ocr import AzureOCR
from src.text_mapper import TextMapper
from src.database import DatabaseManager
from src.llm_field_extractor import LLMFieldExtractor
from config.config import INPUT_DIR, OCR_RESULTS_DIR, VISUALIZATIONS_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_pdf(pdf_path: str, page_num: Optional[int] = None, 
                save_to_db: bool = False, search_terms: Optional[list] = None,
                use_llm: bool = False) -> None:
    """
    Process a PDF file with Azure OCR and optional LLM extraction
    
    Args:
        pdf_path: Path to the PDF file
        page_num: Specific page to process (None = all pages)
        save_to_db: Whether to save results to database
        search_terms: Optional list of terms to search for
        use_llm: Whether to use LLM for intelligent field extraction
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        logger.error(f"❌ PDF file not found: {pdf_path}")
        return
    
    logger.info("="*60)
    logger.info(f"🚀 Starting PDF OCR Processing")
    logger.info(f"   File: {pdf_path.name}")
    if use_llm:
        logger.info(f"   🧠 LLM extraction: ENABLED")
    logger.info("="*60)
    
    try:
        # Step 1: Convert PDF to images
        logger.info("\n📄 Step 1: Converting PDF to images...")
        processor = PDFProcessor()
        images = processor.convert_pdf_to_images(str(pdf_path))
        
        # Get PDF info
        pdf_info = processor.get_pdf_info(str(pdf_path))
        
        # Filter to specific page if requested
        if page_num is not None:
            images = [(num, img) for num, img in images if num == page_num]
            if not images:
                logger.error(f"❌ Page {page_num} not found in PDF")
                return
        
        # Step 2: Initialize Azure OCR
        logger.info("\n🔍 Step 2: Initializing Azure OCR...")
        ocr = AzureOCR()
        
        # Step 3: Initialize LLM (if requested)
        llm_extractor = None
        if use_llm:
            logger.info("\n🧠 Step 3: Initializing LLM Field Extractor...")
            try:
                llm_extractor = LLMFieldExtractor()
            except ValueError as e:
                logger.error(f"❌ {str(e)}")
                logger.info("   Set ANTHROPIC_API_KEY environment variable to use LLM")
                use_llm = False
        
        # Step 4: Initialize database (if needed)
        db = None
        doc_id = None
        if save_to_db:
            logger.info("\n💾 Step 4: Initializing database...")
            db = DatabaseManager()
            if db.engine:
                db.create_tables()
                doc_id = db.save_document(
                    filename=pdf_path.name,
                    file_path=str(pdf_path),
                    total_pages=pdf_info['total_pages'],
                    file_size_mb=pdf_info['file_size_mb'],
                    pdf_metadata=pdf_info['metadata']
                )
        
        # Step 5: Process each page
        logger.info("\n⚙️  Step 5: Processing pages with OCR...")
        
        all_search_results = {}
        all_llm_results = {}
        
        for page_num, image in images:
            logger.info(f"\n   Processing page {page_num}...")
            
            # Run OCR
            results = ocr.process_image(image)
            
            # Save OCR results to JSON
            output_filename = f"{pdf_path.stem}_page_{page_num}_ocr.json"
            output_path = OCR_RESULTS_DIR / output_filename
            ocr.save_results(results, str(output_path))
            
            # Save visualization
            vis_filename = f"{pdf_path.stem}_page_{page_num}.png"
            vis_path = VISUALIZATIONS_DIR / vis_filename
            processor.save_image(image, str(vis_path))
            
            # Save to database
            if db and doc_id:
                db.save_ocr_result(doc_id, page_num, results)
            
            # LLM extraction (if enabled)
            if use_llm and llm_extractor:
                logger.info(f"\n   🧠 Running LLM extraction on page {page_num}...")
                
                extracted = llm_extractor.extract_fields(results)
                enriched = llm_extractor.enrich_with_bounding_boxes(extracted, results)
                
                # Save LLM results
                llm_output_filename = f"{pdf_path.stem}_page_{page_num}_llm.json"
                llm_output_path = OCR_RESULTS_DIR / llm_output_filename
                
                with open(llm_output_path, 'w', encoding='utf-8') as f:
                    json.dump(enriched, f, indent=2, ensure_ascii=False)
                
                all_llm_results[page_num] = enriched
                
                logger.info(f"   ✅ LLM extracted {enriched['total_fields_found']} fields")
                logger.info(f"      Document Type: {enriched['document_type']}")
                logger.info(f"      Saved to: {llm_output_path}")
            
            # Search for terms if provided
            if search_terms:
                mapper = TextMapper(results)
                page_search_results = mapper.search_multiple(search_terms)
                
                for term, matches in page_search_results.items():
                    if term not in all_search_results:
                        all_search_results[term] = []
                    
                    for match in matches:
                        match['page'] = page_num
                        all_search_results[term].append(match)
            
            # Print summary
            logger.info(f"   ✅ Page {page_num} completed:")
            logger.info(f"      - Text blocks extracted: {len(results['text_blocks'])}")
            logger.info(f"      - OCR results: {output_path}")
            logger.info(f"      - Image saved: {vis_path}")
        
        # Print LLM extraction results
        if use_llm and all_llm_results:
            logger.info("\n" + "="*60)
            logger.info("🧠 LLM EXTRACTION RESULTS:")
            logger.info("="*60)
            
            for page_num, enriched in all_llm_results.items():
                logger.info(f"\n📄 Page {page_num}:")
                logger.info(f"   Document Type: {enriched['document_type']}")
                logger.info(f"   Fields Found: {enriched['total_fields_found']}")
                
                for field_key, field_data in enriched['extracted_fields'].items():
                    logger.info(f"\n   ✅ {field_data['standard_name']}")
                    logger.info(f"      Value: {field_data['value']}")
                    logger.info(f"      Confidence: {field_data['confidence']}")
                    bbox = field_data['bounding_box']
                    logger.info(f"      Position: ({bbox['left']:.0f}, {bbox['top']:.0f})")
        
        # Print search results
        if search_terms and all_search_results:
            logger.info("\n" + "="*60)
            logger.info("🔍 SEARCH RESULTS:")
            for term, matches in all_search_results.items():
                logger.info(f"\n   '{term}': {len(matches)} matches")
                for match in matches[:3]:  # Show first 3
                    bbox = match['bounding_box']
                    logger.info(f"      - Page {match['page']}: Position ({bbox['left']:.0f}, {bbox['top']:.0f})")
        
        logger.info("\n" + "="*60)
        logger.info("🎉 Processing completed successfully!")
        logger.info(f"   Total pages processed: {len(images)}")
        logger.info(f"   Results directory: {OCR_RESULTS_DIR}")
        if save_to_db and doc_id:
            logger.info(f"   Database document ID: {doc_id}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ Error during processing: {str(e)}", exc_info=True)
        sys.exit(1)


def process_existing_ocr_with_llm(ocr_json_path: str) -> None:
    """
    Process an existing OCR JSON file with LLM extraction
    
    Args:
        ocr_json_path: Path to the OCR JSON file
    """
    ocr_json_path = Path(ocr_json_path)
    
    if not ocr_json_path.exists():
        logger.error(f"❌ OCR JSON file not found: {ocr_json_path}")
        return
    
    logger.info("="*60)
    logger.info(f"🧠 Processing existing OCR with LLM")
    logger.info(f"   File: {ocr_json_path.name}")
    logger.info("="*60)
    
    try:
        # Load OCR results
        logger.info("\n📄 Loading OCR results...")
        with open(ocr_json_path, 'r', encoding='utf-8') as f:
            ocr_results = json.load(f)
        
        logger.info(f"   ✅ Loaded {len(ocr_results.get('text_blocks', []))} text blocks")
        
        # Initialize LLM
        logger.info("\n🧠 Initializing LLM Field Extractor...")
        llm_extractor = LLMFieldExtractor()
        
        # Extract fields
        logger.info("\n⚙️  Extracting fields with LLM...")
        extracted = llm_extractor.extract_fields(ocr_results)
        enriched = llm_extractor.enrich_with_bounding_boxes(extracted, ocr_results)
        
        # Save results
        output_path = ocr_json_path.parent / f"{ocr_json_path.stem}_llm.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(enriched, f, indent=2, ensure_ascii=False)
        
        # Print results
        logger.info("\n" + "="*60)
        logger.info("📊 EXTRACTION RESULTS:")
        logger.info("="*60)
        logger.info(f"\n   Document Type: {enriched['document_type']}")
        logger.info(f"   Fields Found: {enriched['total_fields_found']}")
        
        for field_key, field_data in enriched['extracted_fields'].items():
            logger.info(f"\n   ✅ {field_data['standard_name']}")
            logger.info(f"      Value: {field_data['value']}")
            logger.info(f"      Confidence: {field_data['confidence']}")
            bbox = field_data['bounding_box']
            logger.info(f"      Position: ({bbox['left']:.0f}, {bbox['top']:.0f})")
            
        
        logger.info("\n" + "="*60)
        logger.info(f"💾 Results saved to: {output_path}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"\n❌ Error during processing: {str(e)}", exc_info=True)
        sys.exit(1)


def list_input_files() -> None:
    """List all PDF files in the input directory"""
    pdf_files = list(INPUT_DIR.glob("*.pdf"))
    
    if not pdf_files:
        logger.info(f"📂 No PDF files found in: {INPUT_DIR}")
        logger.info(f"   Please place PDF files in the input directory")
        return
    
    logger.info(f"\n📂 PDF files in input directory:")
    for i, pdf in enumerate(pdf_files, 1):
        size_mb = pdf.stat().st_size / (1024 * 1024)
        logger.info(f"   {i}. {pdf.name} ({size_mb:.2f} MB)")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Process PDF files with Azure Computer Vision OCR and LLM extraction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic OCR processing
  python main.py --file data/input/document.pdf
  
  # Process with LLM field extraction
  python main.py --file data/input/document.pdf --llm
  
  # Process specific page with LLM
  python main.py --file data/input/document.pdf --page 1 --llm
  
  # Process existing OCR JSON with LLM
  python main.py --llm-only data/ocr_results/sample_page_1_ocr.json
  
  # Process and save to database
  python main.py --file data/input/document.pdf --db --llm
  
  # Traditional search (without LLM)
  python main.py --file data/input/document.pdf --search "862909" "B3219"
  
  # List all PDF files
  python main.py --list
        """
    )
    
    parser.add_argument(
        '--file', '-f',
        type=str,
        help='Path to PDF file to process'
    )
    
    parser.add_argument(
        '--page', '-p',
        type=int,
        help='Specific page number to process (default: all pages)'
    )
    
    parser.add_argument(
        '--db',
        action='store_true',
        help='Save results to database'
    )
    
    parser.add_argument(
        '--search', '-s',
        nargs='+',
        help='Search terms to look for in OCR results (traditional method)'
    )
    
    parser.add_argument(
        '--llm',
        action='store_true',
        help='Use LLM for intelligent field extraction (requires ANTHROPIC_API_KEY)'
    )
    
    parser.add_argument(
        '--llm-only',
        type=str,
        metavar='OCR_JSON',
        help='Process existing OCR JSON file with LLM (skip OCR step)'
    )
    
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='List all PDF files in input directory'
    )
    
    args = parser.parse_args()
    
    # Handle list command
    if args.list:
        list_input_files()
        return
    
    # Handle LLM-only processing
    if args.llm_only:
        process_existing_ocr_with_llm(args.llm_only)
        return
    
    # Handle file processing
    if args.file:
        process_pdf(
            pdf_path=args.file,
            page_num=args.page,
            save_to_db=args.db,
            search_terms=args.search,
            use_llm=args.llm
        )
    else:
        # No arguments - show help
        parser.print_help()
        print("\n")
        list_input_files()


if __name__ == "__main__":
    main()