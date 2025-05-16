def render_tab(neo4j_service, train_asd_detection_model, evaluate_model, reinsert_labels_from_csv):
    import streamlit as st
    import uuid
    import numpy as np

    st.header("🤖 ASD Detection Model")
    if st.button("🔄 Train/Refresh Model"):
        with st.spinner("Training model with leakage protection..."):
            results = train_asd_detection_model(cache_key=str(uuid.uuid4()))
            if results:
                st.session_state.model_results = results
                st.session_state.model_trained = True
                evaluate_model(
                    results["model"],
                    results["X_test"],
                    results["y_test"]
                )
                with st.spinner("Reattaching labels to cases..."):
                    csv_url = "https://raw.githubusercontent.com/GiorgosBouh/llm2cypher-asd-app/main/Toddler_Autism_dataset_July_2018_2.csv"
                    reinsert_labels_from_csv(csv_url)
                    st.success("🎯 Labels reinserted automatically after training!")
            if st.session_state.get("model_trained"):
                st.success("✅ Model trained successfully!")

    with st.expander("🧪 Compare old vs new embeddings (Case 1)"):
        with neo4j_service.session() as session:
            if st.button("📤 Save current embedding of Case 1"):
                result = session.run("MATCH (c:Case {id: 1}) RETURN c.embedding AS emb").single()
                if result and result["emb"]:
                    st.session_state.saved_embedding_case1 = result["emb"]
                    st.success("✅ Saved current embedding of Case 1")

            if st.button("📥 Compare to current embedding of Case 1"):
                result = session.run("MATCH (c:Case {id: 1}) RETURN c.embedding AS emb").single()
                if result and result["emb"]:
                    new_emb = result["emb"]
                    old_emb = st.session_state.get("saved_embedding_case1")
                    if old_emb:
                        diff = np.linalg.norm(np.array(old_emb) - np.array(new_emb))
                        st.write(f"📏 Difference (L2 norm): `{diff:.4f}`")
                        if diff < 1e-3:
                            st.warning("⚠️ Embedding is almost identical.")
                        else:
                            st.success("✅ Embedding changed.")
                    else:
                        st.error("❌ No saved embedding. Click save first.")