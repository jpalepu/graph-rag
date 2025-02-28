from typing import List, Dict, Any
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
from dataclasses import dataclass

@dataclass
class Relationship:
    source: str
    target: str
    type: str

class EntityExtractor:
    def __init__(self):
        self.nlp = spacy.load('en_core_web_lg')
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    def extract_entities(self, text: str) -> List[Any]:
        doc = self.nlp(text)
        return doc.ents

    def get_entity_embeddings(self, entities: List[Any]) -> Dict[str, np.ndarray]:
        embeddings = {}
        for entity in entities:
            embedding = self.embedding_model.encode(entity.text)
            embeddings[entity.text] = embedding
        return embeddings

    def extract_relationships(self, text: str, entities: List[Any]) -> List[Relationship]:
        doc = self.nlp(text)
        relationships = []
        
        entity_spans = {(ent.start, ent.end): ent for ent in entities}
        
        for token in doc:
            if token.dep_ in ('nsubj', 'nsubjpass') and token.head.pos_ == 'VERB':
                for child in token.head.children:
                    if child.dep_ in ('dobj', 'pobj'):
                        source_span = self._find_entity_span(token.i, entity_spans)
                        target_span = self._find_entity_span(child.i, entity_spans)
                        
                        if source_span and target_span:
                            relationships.append(
                                Relationship(
                                    source=entity_spans[source_span].text,
                                    target=entity_spans[target_span].text,
                                    type=token.head.lemma_
                                )
                            )
        
        return relationships

    def _find_entity_span(self, token_idx: int, entity_spans: Dict[tuple, Any]) -> tuple:
        for (start, end), entity in entity_spans.items():
            if start <= token_idx < end:
                return (start, end)
        return None