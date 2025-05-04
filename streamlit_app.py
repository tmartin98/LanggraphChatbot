import streamlit as st
import os
import tempfile
import sys
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langgraph.graph import END, StateGraph
from typing import List, Dict, Any, TypedDict
import uuid


sys.setrecursionlimit(10000)

# Configuration
OLLAMA_MODEL = "llama3.2"  
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# --- Models ---
class AgentState(TypedDict):
    """Model for the agent state"""
    user_input: str
    context: List[str]
    thoughts: List[str]
    planning_steps: List[str]  
    current_step: int
    response: str
    query_refinements: List[str]
    evaluation: str
    conversation_history: List[Dict[str, str]]
    iteration_count: int  

# --- Components ---
class DocumentProcessor:
    """Processing and vectorizing PDF documents"""
    
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.vectorstore = None
    
    def process_pdf(self, pdf_file):
        """Process PDF file and load into vector store"""
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        # Splitting text
        chunks = self.text_splitter.split_text(text)
        
        # Creating vector store
        self.vectorstore = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            persist_directory=f"./chroma_db_{uuid.uuid4()}"
        )
        
        return len(chunks)
    
    def search(self, query, k=3):
        """Search in vector store"""
        if not self.vectorstore:
            return []
        
        results = self.vectorstore.similarity_search(query, k=k)
        return [doc.page_content for doc in results]

# --- Agentic components ---
class TaskPlanner:
    """Task planning component"""
    
    def __init__(self, llm):
        self.llm = llm
        self.template = """
        You are a smart Task Planner who breaks down user questions into subtasks.
        
        User question: {user_input}
        
        Break down answering this question into steps. These steps should include:
        1. Interpreting and clarifying the question
        2. Identifying necessary information
        3. Determining search strategy
        4. Extracting relevant information
        5. Composing the answer
        
        Provide the steps in exact order.
        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["user_input"])
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def run(self, state: AgentState) -> AgentState:
        """Create plan"""
        result = self.chain.run(user_input=state["user_input"])
        state["thoughts"].append(f"Creating plan: {result}")
        state["planning_steps"] = [step.strip() for step in result.split("\n") if step.strip()]
        state["current_step"] = 0
        state["iteration_count"] = 0 
        return state

class QueryRefiner:
    """Search query refinement"""
    
    def __init__(self, llm):
        self.llm = llm
        self.template = """
        You are a search expert who helps refine and improve search queries.
        
        Original question: {user_input}
        Plan: {plan}
        Current step: {current_step}
        
        You need to determine search terms needed to execute this step.
        Provide 2-3 different search terms that will help find relevant information.
        Only provide the search terms, one per line.
        """
        self.prompt = PromptTemplate(
            template=self.template, 
            input_variables=["user_input", "plan", "current_step"]
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def run(self, state: AgentState) -> AgentState:
        """Refine search expressions"""
        state["iteration_count"] = state.get("iteration_count", 0) + 1
        
        if state["current_step"] < len(state["planning_steps"]):
            current_step = state["planning_steps"][state["current_step"]]
            result = self.chain.run(
                user_input=state["user_input"], 
                plan="\n".join(state["planning_steps"]), 
                current_step=current_step
            )
            queries = [q.strip() for q in result.split("\n") if q.strip()]
            state["thoughts"].append(f"Search terms for step {state['current_step']+1}: {queries}")
            state["query_refinements"] = queries
        return state

class InformationRetriever:
    """Information extraction component"""
    
    def __init__(self, doc_processor):
        self.doc_processor = doc_processor
    
    def run(self, state: AgentState) -> AgentState:
        """Extract information from vector store"""
        contexts = []
        for query in state["query_refinements"]:
            results = self.doc_processor.search(query)
            contexts.extend(results)
        
        # Remove duplicates
        unique_contexts = list(set(contexts))
        state["thoughts"].append(f"Number of results: {len(unique_contexts)}")
        state["context"] = unique_contexts
        return state

class ResponseGenerator:
    """Response generation component"""
    
    def __init__(self, llm):
        self.llm = llm
        self.template = """
        You are a helpful assistant who responds to user questions based on available information.
        
        User question: {user_input}
        
        The following context information is available:
        {context}
        
        Previous thoughts:
        {thoughts}
        
        Answer the user's question based on the context. If you can't find appropriate information, indicate that you don't have enough information to answer the question.
        Provide a structured, informative response. Don't mention internal processes (thoughts, plans, etc).
        """
        self.prompt = PromptTemplate(
            template=self.template, 
            input_variables=["user_input", "context", "thoughts"]
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def run(self, state: AgentState) -> AgentState:
        """Generate response"""
        context_text = "\n\n".join(state["context"]) if state["context"] else "No context available."
        thoughts_text = "\n".join(state["thoughts"])
        
        result = self.chain.run(
            user_input=state["user_input"],
            context=context_text,
            thoughts=thoughts_text
        )
        
        state["response"] = result
        state["current_step"] += 1
        state["thoughts"].append(f"Completed step {state['current_step']} of {len(state['planning_steps'])}")
        return state

class SelfEvaluator:
    """Self-evaluation component"""
    
    def __init__(self, llm):
        self.llm = llm
        self.template = """
        You are a critical evaluator who analyzes the quality of responses.
        
        Original question: {user_input}
        Available context: {context}
        Generated response: {response}
        
        Evaluate the response based on the following criteria:
        1. Accuracy - Does the response match the available information?
        2. Completeness - Does the response address all relevant points?
        3. Clarity - Is the response understandable and well-structured?
        
        If you find any deficiencies, briefly suggest how to improve.
        """
        self.prompt = PromptTemplate(
            template=self.template, 
            input_variables=["user_input", "context", "response"]
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt)
    
    def run(self, state: AgentState) -> AgentState:
        """Evaluate response"""
        context_text = "\n\n".join(state["context"]) if state["context"] else "No context available."
        
        result = self.chain.run(
            user_input=state["user_input"],
            context=context_text,
            response=state["response"]
        )
        
        state["evaluation"] = result
        state["thoughts"].append(f"Evaluation: {result}")
        return state

# --- Decision logic ---
def should_continue_planning(state: AgentState) -> str:
    """Decides whether to continue planning or finish"""
    # Hard limit on iterations to prevent infinite recursion
    if state.get("iteration_count", 0) >= 10:
        state["thoughts"].append("Reached maximum iterations, finishing.")
        return "complete"
    
    # Check if we've gone through all steps
    if state["current_step"] >= len(state["planning_steps"]):
        state["thoughts"].append("Completed all steps in plan, finishing.")
        return "complete"
    
    state["thoughts"].append(f"Continuing to next step: {state['current_step']+1}/{len(state['planning_steps'])}")
    return "continue"

# --- LangGraph integration ---
def create_agent_graph(doc_processor):
    """Create agent graph"""
    llm = Ollama(model=OLLAMA_MODEL)
    
    # Initialize components
    task_planner = TaskPlanner(llm)
    query_refiner = QueryRefiner(llm)
    info_retriever = InformationRetriever(doc_processor)
    response_generator = ResponseGenerator(llm)
    self_evaluator = SelfEvaluator(llm)
    
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("planning", task_planner.run)
    workflow.add_node("refine_query", query_refiner.run)
    workflow.add_node("retrieve", info_retriever.run)
    workflow.add_node("generate", response_generator.run)
    workflow.add_node("evaluate", self_evaluator.run)
    
    # Add edges
    workflow.add_edge("planning", "refine_query")
    workflow.add_edge("refine_query", "retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", "evaluate")
    
    # Add branching
    workflow.add_conditional_edges(
        "evaluate",
        should_continue_planning,
        {
            "continue": "refine_query",
            "complete": END
        }
    )
    
    # Set entry point
    workflow.set_entry_point("planning")
    
    return workflow.compile()

# --- Streamlit UI ---
def main():
    st.set_page_config(page_title="Agentic RAG Chatbot", page_icon="🤖", layout="wide")
    
    st.title("Agentic RAG Chatbot")
    st.write("""
    This is a prototype application that demonstrates agentic RAG (Retrieval-Augmented Generation).
    Upload a PDF document and ask questions about it!
    """)
    
    # Initialization
    if "doc_processor" not in st.session_state:
        st.session_state.doc_processor = DocumentProcessor()
        st.session_state.uploaded_file = None
        st.session_state.agent = None
        st.session_state.conversation = []
    
    # Sidebar - PDF upload
    with st.sidebar:
        st.header("Document Upload")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file and (not st.session_state.uploaded_file or 
                              st.session_state.uploaded_file != uploaded_file.name):
            with st.spinner("Processing document..."):
                # Save file to temporary location
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_path = tmp_file.name
                
                # Processing
                chunk_count = st.session_state.doc_processor.process_pdf(tmp_path)
                st.session_state.uploaded_file = uploaded_file.name
                
                # Create agent
                st.session_state.agent = create_agent_graph(st.session_state.doc_processor)
                
                # Delete temporary file
                os.unlink(tmp_path)
                
            st.success(f"Document successfully processed! ({chunk_count} segments)")
    
    # Chat area
    st.header("Chat")
    
    # Display previous conversation
    for message in st.session_state.conversation:
        with st.chat_message(message["role"]):
            st.write(message["content"])
    
    # User input
    user_input = st.chat_input("Ask something about the document...")
    
    if user_input and st.session_state.agent:
        # Display user message
        with st.chat_message("user"):
            st.write(user_input)
        st.session_state.conversation.append({"role": "user", "content": user_input})
        
        # Run agent
        with st.spinner("Thinking..."):
            initial_state = AgentState(
                user_input=user_input,
                context=[],
                thoughts=[],
                planning_steps=[], 
                current_step=0,
                response="",
                query_refinements=[],
                evaluation="",
                conversation_history=st.session_state.conversation.copy(),
                iteration_count=0 
            )
            
            result = st.session_state.agent.invoke(
                initial_state,
                {"recursion_limit": 50}  
            )
            
            # Debug information
            with st.expander("Agent operation details"):
                st.subheader("Plan")
                for i, step in enumerate(result["planning_steps"]):
                    st.write(f"{i+1}. {step}")
                
                st.subheader("Thoughts")
                for thought in result["thoughts"]:
                    st.write(f"- {thought}")
                
                st.subheader("Evaluation")
                st.write(result["evaluation"])
                
                st.subheader("Metrics")
                st.write(f"Iterations: {result.get('iteration_count', 'N/A')}")
            
        # Display response
        with st.chat_message("assistant"):
            st.write(result["response"])
        st.session_state.conversation.append({"role": "assistant", "content": result["response"]})
    
    elif user_input and not st.session_state.agent:
        with st.chat_message("assistant"):
            st.write("Please upload a PDF document first!")
        st.session_state.conversation.append({"role": "assistant", "content": "Please upload a PDF document first!"})

if __name__ == "__main__":
    main()
