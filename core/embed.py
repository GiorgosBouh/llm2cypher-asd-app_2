
import os
import networkx as nx
from node2vec import Node2Vec

def generate_embeddings(driver):
    G = nx.Graph()
    with driver.session() as session:
        result = session.run("""
            MATCH (c:Case)
            OPTIONAL MATCH (c)-[r:HAS_ANSWER|HAS_DEMOGRAPHIC|SUBMITTED_BY|SIMILAR_TO]->(other)
            WHERE c.id IS NOT NULL AND other.id IS NOT NULL
            RETURN c.id AS node_id, collect(DISTINCT other.id) AS neighbors
        """)
        for record in result:
            node_id = str(record["node_id"])
            G.add_node(node_id)
            for neighbor in record["neighbors"]:
                G.add_edge(node_id, str(neighbor))

    if len(G.nodes) < 10:
        raise ValueError(f"❌ Not enough nodes ({len(G.nodes)}) for meaningful embeddings")

    print(f"📊 Graph stats: {len(G.nodes)} nodes, {len(G.edges)} edges")

    node2vec = Node2Vec(
        G,
        dimensions=128,
        walk_length=30,
        num_walks=200,
        workers=4,
        p=1.0,
        q=0.5,
        temp_folder=os.path.join(os.getcwd(), 'node2vec_temp')
    )

    model = node2vec.fit(window=10, min_count=1, batch_words=128)

    with driver.session() as session:
        batch = []
        for node_id in G.nodes():
            emb = model.wv[str(node_id)].tolist()
            batch.append({"node_id": int(node_id), "embedding": emb})
            if len(batch) >= 1000:
                session.run("""
                    UNWIND $batch AS item
                    MATCH (c:Case {id: item.node_id})
                    SET c.embedding = item.embedding
                """, {"batch": batch})
                batch = []
        if batch:
            session.run("""
                UNWIND $batch AS item
                MATCH (c:Case {id: item.node_id})
                SET c.embedding = item.embedding
            """, {"batch": batch})

    print(f"✅ Successfully saved embeddings for {len(G.nodes)} nodes")
    return True
