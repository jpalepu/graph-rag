from typing import Dict, List, Any, Tuple
from neo4j import GraphDatabase
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from openai import AzureOpenAI
import os
import logging

class QueryProcessor:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.nlp = spacy.load('en_core_web_lg')
        
        self.client = AzureOpenAI(
            api_key=os.getenv("AZURE_OPENAI_KEY", "1052dad9e27648a489301296eef3f95f"),
            api_version="2023-05-15",
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", "https://ict-gpt.openai.azure.com")
        )

    def _extract_query_entities(self, query: str) -> List[str]:
        doc = self.nlp(query)
        entities = []
        for ent in doc.ents:
            entities.append(ent.text)
        return entities

    def _find_similar_nodes(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query)
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n)
                WHERE n.embedding IS NOT NULL
                WITH n, gds.similarity.cosine(n.embedding, $embedding) AS similarity
                WHERE similarity > 0.5
                RETURN n.name AS entity, labels(n)[0] AS label, similarity
                ORDER BY similarity DESC
                LIMIT $limit
            """, embedding=query_embedding.tolist(), limit=limit)
            
            return [dict(record) for record in result]

    def _find_entity_paths(self, entities: List[str], max_depth: int = 3) -> List[Dict[str, Any]]:
        paths = []
        with self.driver.session() as session:
            for entity in entities:
                result = session.run("""
                    MATCH path = (start)-[*1..3]-(end)
                    WHERE start.name CONTAINS $entity
                    RETURN path, length(path) AS path_length
                    ORDER BY path_length
                    LIMIT 5
                """, entity=entity)
                
                for record in result:
                    path = record["path"]
                    path_str = self._format_path(path)
                    paths.append({"path": path_str})
        
        return paths

    def _format_path(self, path) -> str:
        nodes = [node["name"] for node in path.nodes]
        rels = [rel.type for rel in path.relationships]
        path_str = nodes[0]
        for i in range(len(rels)):
            path_str += f" --[{rels[i]}]--> {nodes[i+1]}"
        return path_str

    def _format_context(self, context: Dict[str, Any], query_entities: List[str]) -> str:
        prompt = "Based on the following information from our knowledge graph:\n\n"

        if context['entity_paths']:
            prompt += "Relationships found:\n"
            for path in context['entity_paths']:
                if 'path' in path:
                    prompt += f"- {path['path']}\n"
                else:
                    prompt += f"- {path['source']} is {path['relationship_type']} to {path['connected_entity']} ({path['connected_label']})\n"

        if context['similar_nodes']:
            prompt += "\nRelated entities:\n"
            for node in context['similar_nodes']:
                if node['entity'] not in query_entities:
                    prompt += f"- {node['entity']} (type: {node['label']})\n"

        return prompt

    def process_query(self, query: str) -> Dict[str, Any]:
        try:
            entities = self._extract_query_entities(query)
            similar_nodes = self._find_similar_nodes(query)
            entity_paths = self._find_entity_paths(entities)
            
            context = {
                'entity_paths': entity_paths,
                'similar_nodes': similar_nodes
            }
            
            formatted_context = self._format_context(context, entities)
            
            messages = [
                {
                    "role": "system",
                    "content": "You are a knowledgeable assistant that helps answer questions based on information from a knowledge graph."
                },
                {
                    "role": "user",
                    "content": f"{formatted_context}\n\nQuestion: {query}\nPlease provide a detailed answer based on the knowledge graph information."
                }
            ]
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            
            return {
                'query': query,
                'context': context,
                'answer': response.choices[0].message.content,
                'entities': entities
            }
            
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            raise

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()