def render_tab():
    import streamlit as st
    import subprocess
    import sys

    st.header("🌐 Graph Embeddings")
    st.warning("⚠️ Developer use only")
    if st.button("🔁 Recalculate All Embeddings"):
        with st.spinner("Running kg_builder_2.py..."):
            result = subprocess.run(
                [sys.executable, "kg_builder_2.py"],
                capture_output=True,
                text=True
            )
            if result.returncode == 0:
                st.success("✅ Embeddings updated!")
            else:
                st.error("❌ Error running script")
                st.code(result.stderr)