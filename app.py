import os
import pypdf
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="Zyro Dynamics HR Help Desk", page_icon="🏢", layout="centered")

def get_secret(key):
    try:
        return st.secrets[key]
    except:
        return os.environ.get(key, "")

GROQ_API_KEY = get_secret("GROQ_API_KEY")
LANGCHAIN_API_KEY = get_secret("LANGCHAIN_API_KEY")

if GROQ_API_KEY:
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
if LANGCHAIN_API_KEY:
    os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = "zyro-rag-challenge"

CORPUS_PATH = "./hr_docs/"

@st.cache_resource(show_spinner="Loading HR policy documents...")
def build_pipeline():
    documents = []
    pdf_files = sorted([f for f in os.listdir(CORPUS_PATH) if f.lower().endswith(".pdf")])
    for pdf_file in pdf_files:
        full_path = os.path.join(CORPUS_PATH, pdf_file)
        reader = pypdf.PdfReader(full_path)
        for page_num, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                documents.append(Document(
                    page_content=text.strip(),
                    metadata={"source": full_path, "filename": pdf_file, "page": page_num}
                ))
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    vectorstore = FAISS.from_documents(chunks, embeddings)
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "fetch_k": 20, "lambda_mult": 0.7},
    )
    from langchain_groq import ChatGroq
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, max_tokens=512)
    return retriever, llm, len(chunks)

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are an HR Help Desk assistant for Zyro Dynamics Pvt. Ltd.
Answer using ONLY the HR policy documents provided below.
If the answer is not in the context, say: "I do not have that information. Please contact hr@zyrodyn.com"

HR Policy Context:
{context}

Employee Question: {question}

Answer:""")

OOS_PROMPT = ChatPromptTemplate.from_template("""
Is this question related to HR topics like leave, salary, WFH, performance, conduct, onboarding, POSH, travel expenses?
Answer ONLY with YES or NO.

Question: {question}
Answer:""")

REFUSAL = "I can only answer HR-related questions about Zyro Dynamics policies. Your question is outside my scope. Is there an HR policy question I can help you with?"

def format_docs(docs):
    parts = []
    for i, doc in enumerate(docs, 1):
        name = doc.metadata.get("filename", "").replace(".pdf", "")
        parts.append(f"[Source {i}: {name}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)

def chat(question, retriever, llm):
    guard = OOS_PROMPT.invoke({"question": question})
    cls = StrOutputParser().invoke(llm.invoke(guard)).strip().upper()
    if "NO" in cls:
        return {"answer": REFUSAL, "sources": [], "refused": True}
    docs = retriever.invoke(question)
    context = format_docs(docs)
    prompt = RAG_PROMPT.invoke({"context": context, "question": question})
    answer = StrOutputParser().invoke(llm.invoke(prompt))
    sources = list({d.metadata.get("filename", "") for d in docs})
    return {"answer": answer, "sources": sources, "refused": False}

retriever, llm, num_chunks = build_pipeline()

st.sidebar.title("Zyro Dynamics HR Help Desk")
st.sidebar.markdown(f"""
Ask questions about:
- Leave policies
- Work from home
- Compensation and benefits
- Performance reviews
- Code of conduct
- IT and data security
- Travel and expenses
- Onboarding and separation

*{num_chunks} policy chunks indexed*
""")
if st.sidebar.button("Clear Chat"):
    st.session_state.messages = st.session_state.messages[:1]
    st.rerun()

st.title("Zyro Dynamics HR Help Desk")
st.caption("Ask any HR policy question — powered by RAG")

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "Hello! I am Zyro HR Help Desk assistant. Ask me anything about our HR policies!",
        "sources": [],
    }]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(f"- {s}")

if user_input := st.chat_input("Type your HR question here..."):
    st.session_state.messages.append({"role": "user", "content": user_input, "sources": []})
    with st.chat_message("user"):
        st.markdown(user_input)
    with st.chat_message("assistant"):
        with st.spinner("Searching HR policies..."):
            result = chat(user_input, retriever, llm)
        st.markdown(result["answer"])
        if result["sources"] and not result["refused"]:
            with st.expander("Sources"):
                for s in result["sources"]:
                    st.markdown(f"- {s}")
        st.caption("Out of scope" if result["refused"] else "Answered from HR policy documents")
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
    })
