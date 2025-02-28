from typing import Dict, List, Any
from neo4j import GraphDatabase
import networkx as nx
from community import community_louvain
import logging
import numpy as np

class KnowledgeGraphBuilder:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))
        self._init_graph()

    def _init_graph(self):
        with self.driver.session() as session:
            session.run("""
                CREATE CONSTRAINT unique_entity_name IF NOT EXISTS
                FOR (n:Entity) REQUIRE n.name IS UNIQUE
            """)

    def create_entity_node(self, entity, embedding: np.ndarray):
        with self.driver.session() as session:
            session.run("""
                MERGE (n:Entity {name: $name})
                SET n.label = $label,
                    n.embedding = $embedding
            """, name=entity.text, label=entity.label_, embedding=embedding.tolist())

    def create_relationship(self, relationship):
        with self.driver.session() as session:
            session.run("""
                MATCH (source:Entity {name: $source_name})
                MATCH (target:Entity {name: $target_name})
                MERGE (source)-[r:RELATED {type: $rel_type}]->(target)
            """, source_name=relationship.source, target_name=relationship.target, rel_type=relationship.type)

    def detect_communities(self) -> Dict[str, int]:
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n:Entity)
                OPTIONAL MATCH (n)-[r]-(m:Entity)
                RETURN n.name AS source, 
                       m.name AS target, 
                       type(r) AS relationship
            """)
            
            G = nx.Graph()
            for record in result:
                source = record["source"]
                target = record["target"]
                if source and target:
                    G.add_edge(source, target)
                elif source:
                    G.add_node(source)

            communities = community_louvain.best_partition(G)
            
            for node, community in communities.items():
                session.run("""
                    MATCH (n:Entity {name: $name})
                    SET n.community = $community
                """, name=node, community=community)
            
            return communities

    def clear_graph(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def close(self):
        self.driver.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()