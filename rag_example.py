from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate

llm = ChatOpenAI(model="gpt-4o-mini")

# PDF loading
loader = PyPDFLoader(file_path="ChatGPT - skrocony kurs B2C PL.pdf")
docs = loader.load()

# Split docs
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

# Vector store
vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())

# Create retriever
retriever = vectorstore.as_retriever()

# Prompt template
prompt = ChatPromptTemplate([
    ("human", "Use the following pieces of retrieved context to answer the question. If you don't know the answer, just sat that you don't know. Question: {question} Context: {context} Answer:")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

response = rag_chain.invoke("Jaka jest cena i ile trwa kurs? Napisz to w formie zartu")

print(response)