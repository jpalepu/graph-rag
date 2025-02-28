import PyPDF2
from typing import List, Dict, Any
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredMarkdownLoader
)
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentProcessor:
    def __init__(self, 
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 length_function = len,
                 max_chunks_per_doc: int = None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.max_chunks_per_doc = max_chunks_per_doc
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
    def _check_file_exists(self, file_path: str) -> bool:
        if not os.path.exists(file_path):
            logging.error(f"File does not exist: {file_path}")
            return False
        if not os.path.isfile(file_path):
            logging.error(f"Path is not a file: {file_path}")
            return False
        if not os.access(file_path, os.R_OK):
            logging.error(f"File is not readable: {file_path}")
            return False
        return True

    def _load_pdf(self, file_path: str) -> str:
        try:
            if not self._check_file_exists(file_path):
                raise ValueError(f"File not accessible: {file_path}")
                
            logging.info(f"Loading PDF file: {file_path}")
            loader = PyPDFLoader(file_path)
            pages = loader.load()
            text = "\n".join([page.page_content for page in pages])
            logging.info(f"Successfully loaded PDF: {file_path}")
            return text
        except Exception as e:
            logging.error(f"Error loading PDF {file_path}: {str(e)}")
            raise

    def _load_text(self, file_path: str) -> str:
        try:
            if not self._check_file_exists(file_path):
                raise ValueError(f"File not accessible: {file_path}")
                
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
            logging.info(f"Successfully loaded text file: {file_path}")
            return text
        except Exception as e:
            logging.error(f"Error loading text file {file_path}: {str(e)}")
            raise

    def _load_markdown(self, file_path: str) -> str:
        try:
            if not self._check_file_exists(file_path):
                raise ValueError(f"File not accessible: {file_path}")
                
            loader = UnstructuredMarkdownLoader(file_path)
            data = loader.load()
            text = "\n".join([doc.page_content for doc in data])
            logging.info(f"Successfully loaded markdown file: {file_path}")
            return text
        except Exception as e:
            logging.error(f"Error loading markdown file {file_path}: {str(e)}")
            raise

    def _chunk_text(self, text: str) -> List[str]:
        try:
            chunks = self.text_splitter.split_text(text)
            
            if self.max_chunks_per_doc and len(chunks) > self.max_chunks_per_doc:
                chunks = chunks[:self.max_chunks_per_doc]
                logging.warning(f"Number of chunks limited to {self.max_chunks_per_doc}")
            
            logging.info(f"Text split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logging.error(f"Error chunking text: {str(e)}")
            raise

    def process_document(self, file_path: str) -> List[str]:
        logging.info(f"Processing document: {file_path}")
        
        file_extension = os.path.splitext(file_path)[1].lower()
        
        try:
            if not self._check_file_exists(file_path):
                raise ValueError(f"File not accessible: {file_path}")
                
            if file_extension == '.pdf':
                text = self._load_pdf(file_path)
            elif file_extension == '.md':
                text = self._load_markdown(file_path)
            else:
                text = self._load_text(file_path)
            
            chunks = self._chunk_text(text)
            
            logging.info(f"Document processing complete. Generated {len(chunks)} chunks.")
            return chunks
            
        except Exception as e:
            logging.error(f"Error processing document {file_path}: {str(e)}")
            raise

    def process_directory(self, directory_path: str) -> Dict[str, List[str]]:
        logging.info(f"Processing directory: {directory_path}")
        
        if not os.path.exists(directory_path):
            raise ValueError(f"Directory does not exist: {directory_path}")
        
        supported_extensions = ['.txt', '.pdf', '.md']
        all_chunks = {}
        
        try:
            files = [
                f for f in os.listdir(directory_path)
                if os.path.splitext(f)[1].lower() in supported_extensions
            ]
            
            for file in tqdm(files, desc="Processing files"):
                file_path = os.path.join(directory_path, file)
                try:
                    chunks = self.process_document(file_path)
                    all_chunks[file] = chunks
                except Exception as e:
                    logging.error(f"Error processing {file}: {str(e)}")
                    continue
            
            logging.info(f"Directory processing complete. Processed {len(all_chunks)} files.")
            return all_chunks
            
        except Exception as e:
            logging.error(f"Error processing directory {directory_path}: {str(e)}")
            raise

if __name__ == "__main__":
    processor = DocumentProcessor(
        chunk_size=1000,
        chunk_overlap=200,
        max_chunks_per_doc=None
    )
    
    chunks = processor.process_document("path/to/your/document.pdf")
    print(f"Generated {len(chunks)} chunks")
    
    all_chunks = processor.process_directory("path/to/your/documents")