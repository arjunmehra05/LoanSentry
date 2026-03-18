import re
import streamlit as st
from utils.retriever import retrieve


def render(rag, groq_api_key=""):
    st.markdown('<div class="section-header">Knowledge Base Search</div>',
                unsafe_allow_html=True)

    query = st.text_input(
        "Search query",
        placeholder="e.g. high DTI ratio risk, debt consolidation default rate")
    top_k = st.slider("Number of results", 1, 5, 5)

    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching knowledge base..."):
            # Clean conversational words for better retrieval
            clean_query = re.sub(
                r'^(what is|what are|how does|how do|tell me about|explain|describe)\s+',
                '', query.lower()).strip()
            results = retrieve(clean_query, rag, top_k=top_k)

        # ── Answer ────────────────────────────────────────────────────────
        st.markdown('<div class="section-header">Answer</div>',
                    unsafe_allow_html=True)

        context = "\n\n".join([
            f"[{r['source']}]\n{r['text'][:600]}"
            for r in results
        ])

        if groq_api_key:
            try:
                from groq import Groq
                client = Groq(api_key=groq_api_key)
                response = client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a credit risk analyst assistant. "
                                "Answer questions based only on the provided "
                                "policy documents. Be concise, specific, and "
                                "use numbers and percentages where available. "
                                "Answer in 3-5 sentences maximum. "
                                "Do not mention document names or sources."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Question: {query}\n\nPolicy Documents:\n{context}"
                        }
                    ],
                    max_tokens=300,
                    temperature=0.2
                )
                st.markdown(response.choices[0].message.content)
            except Exception as e:
                st.error(f"Groq error: {e}")
        else:
            st.warning("Add GROQ_API_KEY to .env to enable AI answers.")

        # ── Source documents ──────────────────────────────────────────────
        st.markdown('<div class="section-header">Source Documents</div>',
                    unsafe_allow_html=True)
        st.success(f"✅ Based on {len(results)} relevant documents")
        for r in results:
            _render_doc(r)

    if st.session_state.get("last_docs"):
        st.markdown(
            '<div class="section-header">Documents Used in Last Assessment</div>',
            unsafe_allow_html=True)
        for doc in st.session_state["last_docs"]:
            _render_doc(doc)


def _render_doc(doc):
    relevance    = max(0, 100 - doc.get("distance", 0) * 50)
    source_clean = (doc["source"]
                    .replace(".txt", "")
                    .replace("_", " ")
                    .title())

    label = f"📄 {source_clean}"
    if "distance" in doc:
        label += f" - Relevance: {relevance:.0f}%"

    with st.expander(label, expanded=False):
        for line in doc["text"].strip().split("\n"):
            line = line.strip()
            if not line:
                st.markdown("")
                continue
            if line.isupper():
                st.markdown(f"#### {line.title()}")
            elif line.startswith("-"):
                st.markdown(f"• {line[1:].strip()}")
            elif len(line) > 1 and line[0].isdigit() and "." in line[:3]:
                st.markdown(f"**{line}**")
            elif ":" in line and len(line.split(":")[0].split()) <= 4:
                parts = line.split(":", 1)
                st.markdown(f"**{parts[0].strip()}:** {parts[1].strip()}")
            else:
                st.write(line)