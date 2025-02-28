import os
from pathlib import Path
from typing import Union
from dotenv import load_dotenv

from data_ingestion import DocumentProcessor
from entity_extraction import EntityExtractor
from graph_builder import KnowledgeGraphBuilder
from query_processor import QueryProcessor


"""
For execution, create the environment file and then use the Neo4j to run) (done with neo4j desktop instead of docker) 
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password_here
"""

class GraphRAG:
    def __init__(self, neo4j_uri: str, neo4j_user: str, neo4j_password: str):
        """Initialize the Graph RAG system."""
        self.document_processor = DocumentProcessor()
        self.entity_extractor = EntityExtractor()
        self.graph_builder = KnowledgeGraphBuilder(
            neo4j_uri, neo4j_user, neo4j_password
        )
        self.query_processor = QueryProcessor(
            neo4j_uri, neo4j_user, neo4j_password
        )

    def process_document(self, file_path: Union[str, Path]):

        """Process a document and build the knowledge graph."""
        # Step 1: Chunk the document
        chunks = self.document_processor.process_document(file_path)
        print(f"Generated {len(chunks)} chunks")

        # Step 2: Extract entities and relationships
        for chunk in chunks:
            # Extract entities
            entities = self.entity_extractor.extract_entities(chunk)
            
            # Generate embeddings
            embeddings = self.entity_extractor.get_entity_embeddings(entities)
            
            # Add entities to graph
            for entity in entities:
                self.graph_builder.create_entity_node(
                    entity,
                    embeddings[entity.text]
                )
            
            # Extract and add relationships
            relationships = self.entity_extractor.extract_relationships(
                chunk,
                entities
            )
            for rel in relationships:
                self.graph_builder.create_relationship(rel)

        # Step 3: Detect communities
        communities = self.graph_builder.detect_communities()
        print(f"Detected {len(set(communities.values()))} communities")

    def query(self, question: str) -> dict:
        """Process a query and return the answer."""
        return self.query_processor.process_query(question)

    def close(self):
        """Close all connections."""
        self.graph_builder.close()
        self.query_processor.close()

def main():
    # Load environment variables
    load_dotenv()
    
    # Get Neo4j credentials from environment variables
    neo4j_uri = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password = os.getenv("NEO4J_PASSWORD", "password")

    # Initialize GraphRAG
    graph_rag = GraphRAG(neo4j_uri, neo4j_user, neo4j_password)

    try:
        # Process document
        document_path = "path/to/your/document.pdf"  # Replace with actual path
        graph_rag.process_document(document_path)

        # Example query
        question = "How does quantum computing affect cryptography?"
        result = graph_rag.query(question)

        print("\nQuestion:", question)
        print("Answer:", result["answer"])
        print("\nContext used:", result["context"])

    finally:
        graph_rag.close()

if __name__ == "__main__":
    main()