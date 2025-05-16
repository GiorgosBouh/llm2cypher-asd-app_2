from neo4j import GraphDatabase
from contextlib import contextmanager
import logging
import streamlit as st
import os
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

class Neo4jService:
    def __init__(self, uri: str, user: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))

    @contextmanager
    def session(self):
        with self.driver.session() as session:
            yield session

    def close(self):
        self.driver.close()

def get_neo4j_service():
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USER")
    pwd = os.getenv("NEO4J_PASSWORD")
    return Neo4jService(uri, user, pwd)

def safe_neo4j_operation(func):
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            st.error(f"Neo4j operation failed: {str(e)}")
            logger.error(f"Neo4j operation failed: {str(e)}")
            return None
    return wrapper
