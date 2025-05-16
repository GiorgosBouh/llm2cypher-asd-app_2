
import streamlit as st
from services.neo4j_service import get_neo4j_service
from utils.nlp import nl_to_cypher
from logic.insert_case import insert_user_case
from graph.embedding import call_embedding_generator, extract_user_embedding
from ml.model import train_asd_detection_model, evaluate_model
from ml.anomaly import train_isolation_forest
from logic.remove_labels import reinsert_labels_from_csv

from streamlit_tabs import tab_model, tab_embedding, tab_upload, tab_nlp

neo4j_service = get_neo4j_service()

def main():
    st.set_page_config(page_title="NeuroCypher ASD", layout="wide")
    st.title("🧠 NeuroCypher ASD")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Model Training",
        "🌐 Graph Embeddings",
        "📤 Upload New Case",
        "💬 NLP to Cypher"
    ])

    with tab1:
        tab_model.render_tab(neo4j_service, train_asd_detection_model, evaluate_model, reinsert_labels_from_csv)
    with tab2:
        tab_embedding.render_tab()
    with tab3:
        tab_upload.render_tab(neo4j_service, insert_user_case, call_embedding_generator, extract_user_embedding, train_isolation_forest)
    with tab4:
        tab_nlp.render_tab(neo4j_service, nl_to_cypher)

if __name__ == "__main__":
    main()
