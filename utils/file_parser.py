import os
from PyPDF2 import PdfReader
import logging

logger = logging.getLogger(__name__)

class FileParsingError(Exception):
    """Custom exception for file parsing errors."""
    pass

def extract_text_from_file(file_path: str) -> str:
    """
    Extracts text content from a file based on its extension.
    Supports .txt and .pdf files.

    Args:
        file_path: The path to the file.

    Returns:
        The extracted text content as a string.

    Raises:
        FileNotFoundError: If the file does not exist.
        FileParsingError: If the file type is unsupported or extraction fails.
    """
    if not os.path.exists(file_path):
        logger.error(f"File not found at path: {file_path}")
        raise FileNotFoundError(f"File not found: {file_path}")

    _, file_extension = os.path.splitext(file_path)
    file_extension = file_extension.lower()

    logger.info(f"Attempting to extract text from {file_path} (type: {file_extension})")

    try:
        if file_extension == '.txt':
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        elif file_extension == '.pdf':
            text = ''
            with open(file_path, 'rb') as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n" # Add newline between pages
            if not text:
                 logger.warning(f"No text could be extracted from PDF: {file_path}. It might be image-based or corrupted.")
                 # Depending on requirements, could raise error or return empty
                 # raise FileParsingError(f"No text could be extracted from PDF: {file_path}. OCR might be needed.")
            return text
        # Placeholder for image OCR
        # elif file_extension in ['.jpg', '.jpeg', '.png', '.tiff']:
        #     logger.info(f"Image file detected. OCR processing needed (not implemented yet).")
        #     # Add OCR logic here (e.g., using pytesseract)
        #     raise FileParsingError(f"OCR for {file_extension} files is not implemented yet.")
        else:
            logger.warning(f"Unsupported file type: {file_extension} for file {file_path}")
            raise FileParsingError(f"Unsupported file type: {file_extension}. Cannot extract text.")
            
    except FileNotFoundError: # Should be caught above, but as safety
        logger.exception(f"File disappeared during processing: {file_path}")
        raise
    except Exception as e:
        logger.exception(f"Failed to extract text from {file_path}. Error: {e}")
        raise FileParsingError(f"Failed to process file {os.path.basename(file_path)}: {e}") from e 