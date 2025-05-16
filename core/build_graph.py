
import sys
import traceback
from neo4j import GraphDatabase
from core.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD, DATASET_URL
from core.parse import parse_csv
from core.nodes import create_nodes
from core.relationships import create_relationships, create_similarity_relationships
from core.embed import generate_embeddings

def connect_to_neo4j(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD):
    print(f"🌐 Connecting to Neo4j: {uri}", flush=True)
    return GraphDatabase.driver(uri, auth=(user, password))

def build_graph():
    driver = connect_to_neo4j()
    try:
        df = parse_csv(DATASET_URL)
        print("🧠 First row:", df.iloc[0].to_dict(), flush=True)

        with driver.session() as session:
            print("🧹 Clearing existing nodes and relationships...", flush=True)
            session.run("MATCH (n) DETACH DELETE n")
            print("⏳ Creating nodes...", flush=True)
            session.execute_write(create_nodes, df)
            print("⏳ Creating relationships...", flush=True)
            session.execute_write(create_relationships, df)
            print("⏳ Creating similarity relationships...", flush=True)
            session.execute_write(create_similarity_relationships, df)

        print("⏳ Generating embeddings...", flush=True)
        generate_embeddings(driver)

        print("✅ Graph construction completed!", flush=True)
        sys.exit(0)

    except Exception as e:
        print(f"❌ Error: {str(e)}", flush=True)
        traceback.print_exc()
        sys.exit(1)

    finally:
        driver.close()
