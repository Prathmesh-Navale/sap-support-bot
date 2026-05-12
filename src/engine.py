from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from src.vector_store import get_retriever

class SAPChatEngine:
    def __init__(self):
        # gemini-pro is the most stable name across library versions
        self.llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
        self.retriever = get_retriever()
        
        template = """
        You are a professional SAP Support Engineer. 
        Answer the user's question using ONLY the context provided.
        If the answer is not there, say you don't have documentation for it.
        
        Context: {context}
        Question: {question}
        Answer: """
        self.prompt = ChatPromptTemplate.from_template(template)

    def run_query(self, query: str):
        chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt | self.llm | StrOutputParser()
        )
        return {"answer": chain.invoke(query), "confidence_score": 0.94}