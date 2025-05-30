import json
import hashlib
from datetime import datetime, UTC
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
from loguru import logger
from langchain.docstore.document import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv
from tqdm import tqdm

# Configure logging
logger.add(
    "process content/Embedding/logs/embedding.log",
    rotation="100 MB",
    retention="1 month",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

# Add evaluation logging
logger.add(
    "process content/Embedding/logs/embedding_evaluation.log",
    rotation="100 MB",
    retention="1 month",
    level="INFO",
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
)

load_dotenv()

class EmbeddingMetrics:
    """Track and analyze embedding generation metrics."""
    
    def __init__(self):
        self.batch_id = None
        self.start_time = None
        self.metrics = {
            "total_documents": 0,
            "total_tokens": 0,
            "total_cost": 0.0,
            "batch_sizes": [],
            "token_counts": [],
            "embedding_times": [],
            "normalization_times": [],
            "errors": [],
            "source_file_hash": None,
            "embedding_model": "text-embedding-3-small",
            "embedding_dim": 1536,
            "normalized": True
        }
    
    def start_batch(self, source_file: str):
        """Start tracking a new embedding batch."""
        self.batch_id = datetime.now(UTC).isoformat()
        self.start_time = datetime.now(UTC)
        
        # Calculate source file hash
        with open(source_file, 'rb') as f:
            self.metrics["source_file_hash"] = hashlib.sha256(f.read()).hexdigest()
        
        logger.info(f"Starting embedding batch {self.batch_id}")
    
    def end_batch(self):
        """End tracking and log metrics."""
        if self.start_time:
            total_time = (datetime.now(UTC) - self.start_time).total_seconds()
            self.metrics["total_time"] = total_time
            
            # Calculate additional metrics
            self.metrics["avg_batch_size"] = np.mean(self.metrics["batch_sizes"])
            self.metrics["avg_tokens_per_doc"] = np.mean(self.metrics["token_counts"])
            self.metrics["avg_embedding_time"] = np.mean(self.metrics["embedding_times"])
            self.metrics["avg_normalization_time"] = np.mean(self.metrics["normalization_times"])
            
            # Log to evaluation log
            logger.bind(batch_id=self.batch_id).info(
                "Embedding Metrics",
                extra={
                    "metrics": self.metrics,
                    "timestamp": datetime.now(UTC).isoformat()
                }
            )
    
    def track_batch(self, batch_size: int, token_count: int, embedding_time: float, normalization_time: float):
        """Track metrics for a single batch."""
        self.metrics["batch_sizes"].append(batch_size)
        self.metrics["token_counts"].append(token_count)
        self.metrics["embedding_times"].append(embedding_time)
        self.metrics["normalization_times"].append(normalization_time)
        self.metrics["total_tokens"] += token_count
        self.metrics["total_cost"] += (token_count / 1000) * 0.00002  # OpenAI's rate
    
    def track_error(self, error: Exception):
        """Track embedding errors."""
        self.metrics["errors"].append({
            "error": str(error),
            "timestamp": datetime.now(UTC).isoformat()
        })

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_documents(json_path: str) -> List[Document]:
    """Load and validate chunked documents."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        documents = []
        for doc in data:
            # Validate required metadata
            required_fields = ['url', 'section', 'article_index', 'chunk_index', 'token_count']
            if not all(field in doc['metadata'] for field in required_fields):
                logger.warning(f"Document missing required metadata: {doc['metadata']}")
                continue
            
            documents.append(Document(
                page_content=doc['page_content'],
                metadata=doc['metadata']
            ))
        
        logger.info(f"Loaded {len(documents)} documents from {json_path}")
        return documents
    
    except Exception as e:
        logger.error(f"Error loading documents: {str(e)}")
        raise

def initialize_embedding_model() -> OpenAIEmbeddings:
    """Initialize the OpenAI embedding model."""
    try:
        embedding_function = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=None,  # Will use OPENAI_API_KEY from environment
            chunk_size=100  # Process in batches of 100
        )
        logger.info("Initialized OpenAI embedding model")
        return embedding_function
    
    except Exception as e:
        logger.error(f"Error initializing embedding model: {str(e)}")
        raise

def generate_embeddings(
    documents: List[Document],
    embedding_function: OpenAIEmbeddings,
    batch_size: int = 100,
    metrics: Optional[EmbeddingMetrics] = None
) -> List[List[float]]:
    """Generate embeddings with progress tracking and metrics."""
    embeddings = []
    total_docs = len(documents)
    
    try:
        for i in tqdm(range(0, total_docs, batch_size), desc="Generating embeddings"):
            batch_start = datetime.now(UTC)
            batch = documents[i:i + batch_size]
            
            # Generate embeddings for batch
            batch_embeddings = embedding_function.embed_documents(
                [doc.page_content for doc in batch]
            )
            embedding_time = (datetime.now(UTC) - batch_start).total_seconds()
            
            # Calculate token count for batch
            token_count = sum(doc.metadata.get('token_count', 0) for doc in batch)
            
            # Normalize batch embeddings
            norm_start = datetime.now(UTC)
            batch_embeddings = normalize_embeddings(batch_embeddings)
            normalization_time = (datetime.now(UTC) - norm_start).total_seconds()
            
            embeddings.extend(batch_embeddings)
            
            # Track metrics
            if metrics:
                metrics.track_batch(
                    len(batch),
                    token_count,
                    embedding_time,
                    normalization_time
                )
        
        if metrics:
            metrics.metrics["total_documents"] = total_docs
        
        return embeddings
    
    except Exception as e:
        if metrics:
            metrics.track_error(e)
        logger.error(f"Error generating embeddings: {str(e)}")
        raise

def normalize_embeddings(embeddings: List[List[float]]) -> List[List[float]]:
    """Apply L2 normalization to embeddings."""
    try:
        # Convert to numpy array
        embeddings_array = np.array(embeddings)
        
        # Calculate L2 norm
        norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
        # Normalize
        normalized_embeddings = embeddings_array / norms
        
        # Verify normalization
        verification_norms = np.linalg.norm(normalized_embeddings, axis=1)
        if not np.allclose(verification_norms, 1.0, atol=1e-6):
            logger.warning("Some embeddings may not be properly normalized")
        
        return normalized_embeddings.tolist()
    
    except Exception as e:
        logger.error(f"Error normalizing embeddings: {str(e)}")
        raise

def create_vector_store(
    documents: List[Document],
    embeddings: List[List[float]],
    metrics: Optional[EmbeddingMetrics] = None
) -> Chroma:
    """Create and configure the vector store."""
    try:
        # Initialize Chroma with metadata
        vector_store = Chroma(
            collection_name="knowledge_base",
            persist_directory="process content/Embedding/embeddings",
            embedding_function=OpenAIEmbeddings(
                model="text-embedding-3-small",
                chunk_size=100
            ),
            collection_metadata={
                "version": "1.0.0",
                "created_at": datetime.now(UTC).isoformat(),
                "source_file_hash": metrics.metrics["source_file_hash"] if metrics else None,
                "embedding_model": "text-embedding-3-small",
                "normalized": True,
                "total_documents": len(documents),
                "embedding_dim": 1536
            }
        )
        
        # Add documents and embeddings
        vector_store.add_texts(
            texts=[doc.page_content for doc in documents],
            metadatas=[doc.metadata for doc in documents],
            embeddings=embeddings
        )
        # Persistence is handled automatically in this version of Chroma
        
        logger.info(f"Created vector store with {len(documents)} documents")
        
        # Log document count in the collection for debugging
        try:
            collection = vector_store._collection
            count = collection.count()
            logger.info(f"[DEBUG] Vector store contains {count} documents after embedding.")
            print(f"[DEBUG] Vector store contains {count} documents after embedding.")
        except Exception as e:
            logger.error(f"[DEBUG] Error checking vector store count: {str(e)}")
        
        return vector_store
    
    except Exception as e:
        if metrics:
            metrics.track_error(e)
        logger.error(f"Error creating vector store: {str(e)}")
        raise

def main():
    """Main execution function."""
    try:
        # Initialize metrics
        metrics = EmbeddingMetrics()
        chunks_path = "process content/chunking/chunks.json"
        metrics.start_batch(chunks_path)
        
        # Load documents
        documents = load_documents(chunks_path)
        
        # Initialize embedding model
        embedding_function = initialize_embedding_model()
        
        # Generate embeddings
        embeddings = generate_embeddings(
            documents,
            embedding_function,
            batch_size=100,
            metrics=metrics
        )
        
        # Create vector store
        vector_store = create_vector_store(
            documents,
            embeddings,
            metrics=metrics
        )
        
        # Log final metrics
        metrics.end_batch()
        
        logger.info("Embedding process completed successfully")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 