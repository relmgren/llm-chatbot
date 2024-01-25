from langchain_community.llms import Ollama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

def metadata_func(record: dict, metadata: dict) -> dict:
    metadata["title"] = record.get("title")
    metadata["html"] = record.get("html")
    return metadata

file_path='./chat/mathem.json'
data_loader = JSONLoader(file_path=file_path, jq_schema='.content[]', content_key='html', text_content=True, metadata_func=metadata_func)
data = data_loader.load()
#print(data[0].metadata)

embeddings = OllamaEmbeddings()
text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(data)
vectors = FAISS.from_documents(documents, embeddings)

prompt = ChatPromptTemplate.from_template("""Answer the following question based only on the provided context:

<context>
{context}
</context>

Question: {input}""")

llm = Ollama(model="llama2")
document_chain = create_stuff_documents_chain(llm, prompt)

retriever = vectors.as_retriever()
retrieval_chain = create_retrieval_chain(retriever, document_chain)

response = retrieval_chain.invoke({"input": "Hur fungerar frakt hos er?"})

print(response["answer"])
