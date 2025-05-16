
def render_tab(neo4j_service, insert_user_case, call_embedding_generator, extract_user_embedding, train_isolation_forest):
    import streamlit as st
    import pandas as pd
    import uuid
    import numpy as np
    import plotly.express as px
    import os

    st.header("📄 Upload New Case")
    # Shortened for brevity; real content includes all checks, upload, validation, visualization, etc.
    uploaded_file = st.file_uploader("Upload your prepared CSV file", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file, delimiter=";")
        st.dataframe(df)
        if len(df) != 1:
            st.error("Upload exactly one case.")
            return
        row = df.iloc[0]
        upload_id = str(uuid.uuid4())
        with st.spinner("Inserting..."):
            insert_user_case(row, upload_id)
        with st.spinner("Generating embedding..."):
            call_embedding_generator(upload_id)
        embedding = extract_user_embedding(upload_id)
        if embedding is not None:
            st.success("✅ Embedding generated")
            st.write(embedding)
            if "model_results" in st.session_state:
                model = st.session_state.model_results["model"]
                proba = model.predict_proba(embedding)[0][1]
                st.metric("Prediction", "ASD Traits" if proba >= 0.5 else "Typical")
                st.metric("Confidence", f"{proba:.1%}")
            iso_result = train_isolation_forest(cache_key=upload_id)
            if iso_result:
                iso_forest, scaler = iso_result
                score = iso_forest.decision_function(scaler.transform(embedding))[0]
                st.metric("Anomaly Score", f"{score:.3f}")
