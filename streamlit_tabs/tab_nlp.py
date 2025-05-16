
def render_tab(neo4j_service, nl_to_cypher):
    import streamlit as st
    import pandas as pd

    st.header("💬 Natural Language to Cypher")
    question = st.text_input("Ask your question:")
    if question:
        cypher = nl_to_cypher(question)
        if cypher:
            st.code(cypher, language="cypher")
            if st.button("Execute Query"):
                with neo4j_service.session() as session:
                    results = session.run(cypher).data()
                    st.dataframe(pd.DataFrame(results))
