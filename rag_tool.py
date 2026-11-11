from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.tools import tool
from langchain_chroma import Chroma 

KNOWLEDGE_DIR = Path("knowledge")
CHROMA_DIR = "chroma_db" 

def  get_vectore_store():
    embeddings = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")


    if Path(CHROMA_DIR).exists():
        print("Loaded existing index from disk")
        return Chroma(persist_directory=CHROMA_DIR,embedding_function=embeddings)


    pdf_files = sorted(KNOWLEDGE_DIR.glob("*.pdf"))
    if not pdf_files:
        raise FileNotFoundError("Put a least one pdf inside the 'knowledge' folder")

    docs = []
    for pdf in pdf_files:
        docs.extend(PyPDFLoader(str(pdf)).load()) 


    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    chunks = splitter.split_documents(docs)

    print(f"Built new index: {len(pdf_files)} PDF(s), {len(docs)} pages, {len(chunks)} chunks")
    return Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_DIR) 


vectore_store = get_vectore_store()
retriver = vectore_store.as_retriever(search_kwargs={"k":3})

@tool
def seach_document(query:str)->str:
    """Search the knowledge-base PDF documents and return the most relevant passages
    with their source file and page. Use this for questions about the documents' content."""

    results = retriver.invoke(query)

    if not results:
        return "No relevant passages found in the documents." 

    parts = []

    for r in results:
        source = Path(r.metadata.get("source","unknow")).name
        page = r.metadata.get("page")
        page_label = page + 1 if isinstance(page,int) else "?"
        parts.append(f"[{source}, page {page_label}]\n{r.page_content}")
    return "\n\n".join(parts)

if __name__ == "__main__":
    print("Knowledge-base search test. Type 'exit' to quit.\n")
    while True:
        q = input("Search:  ").strip() 
        if q.lower() in ["exit","quit"]:
            break
        if q:
            print(seach_document.invoke({"query":q}))
            print("-" * 50)


     

