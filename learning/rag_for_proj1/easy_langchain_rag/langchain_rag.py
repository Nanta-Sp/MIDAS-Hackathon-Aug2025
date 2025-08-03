"""
Simple LangChain RAG System

Core RAG concepts: Retrieval + Generation + Prompting
Uses OpenAI GPT and FAISS vector store.

Required: pip install langchain langchain-openai langchain-community faiss-cpu
"""

import json
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class SimpleRAG:
    """Simple RAG: Retrieval + Generation + Prompting"""
    
    def __init__(self):
        # Load documents and create vector store
        documents = self._load_documents()
        
        # Set up embeddings and vector store (RETRIEVAL)
        embeddings = OpenAIEmbeddings()
        self.vectorstore = FAISS.from_documents(documents, embeddings)
        
        # Set up LLM (GENERATION)  
        self.llm = OpenAI(temperature=0)
        
        # Create RAG chain with custom prompt (PROMPTING)
        prompt = PromptTemplate(
            template="""Answer the question based on the context below.
            
Context: {context}

Question: {question}

Answer:""",
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
    
    def _load_documents(self):
        """Load and split documents"""
        with open("synthetic_knowledge_base.json", 'r') as f:
            data = json.load(f)
        
        documents = []
        for doc in data['documents']:
            documents.append(Document(
                page_content=f"{doc['title']}\n\n{doc['content']}",
                metadata={"title": doc['title'], "category": doc['category']}
            ))
        
        # Split documents into chunks
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        return splitter.split_documents(documents)
    
    def ask(self, question: str):
        """Ask a question and get an answer with sources"""
        result = self.qa_chain({"query": question})
        
        print(f"\nQ: {question}")
        print(f"A: {result['result']}")
        print("Sources:")
        for i, doc in enumerate(result['source_documents'], 1):
            print(f"  {i}. {doc.metadata['title']}")
        
        return result


# Demo
if __name__ == "__main__":
    rag = SimpleRAG()
    
    # Test questions
    questions = [
        "What is machine learning?",
        "How does web development work?", 
        "Tell me about cybersecurity"
    ]
    
    for question in questions:
        rag.ask(question)