from typing import List, Dict, Any
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_openai import ChatOpenAI
from langchain_community.tools import BaseTool
from langchain.memory import ConversationBufferMemory
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, AIMessage

class BaseAnalysisAgent:
    def __init__(self, 
                 llm_model: str,
                 tools: List[BaseTool],
                 memory: ConversationBufferMemory = None,
                 prompt_template: str = None):
        self.llm = ChatOpenAI(model_name=llm_model, temperature=0.1)
        self.tools = tools
        self.memory = memory or ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        self.prompt_template = prompt_template
        self.agent_executor = self._create_agent()

    def _create_agent(self) -> StateGraph:
        if not self.prompt_template:
            # Default prompt if none provided
            self.prompt_template = """You are a specialized financial analysis agent.
            You have access to tools for analyzing stocks.
            Your goal is to provide accurate and helpful analysis based on the available data.
            
            When analyzing, always:
            1. Use relevant tools to gather necessary data
            2. Consider multiple factors before making conclusions
            3. Provide clear explanations for your analysis
            4. Be objective and data-driven
            5. Make sure to provide all required input parameters for tools
            
            Tools available: {tools}
            
            Previous conversation:
            {chat_history}
            
            Current task:
            {input}
            
            Think through this step-by-step:
            1. What information do I need?
            2. Which tools should I use?
            3. How should I analyze the data?
            
            Your response:"""

        # Create prompt
        prompt = PromptTemplate.from_template(self.prompt_template)
        
        # Define the agent state
        def agent_state():
            return {
                "messages": [],
                "next_step": None,
                "tool_output": None,
                "final_answer": None
            }

        # Create the graph
        workflow = StateGraph(agent_state)
        
        # Define the agent steps
        def should_continue(state):
            if state["final_answer"]:
                return END
            return "agent_step"
            
        def process_step(state):
            messages = state["messages"]
            if state["tool_output"]:
                messages.append(AIMessage(content=str(state["tool_output"])))
            
            # Combine prompt with history
            full_prompt = prompt.format(
                tools=self.tools,
                chat_history=self.memory.chat_memory.messages,
                input=messages[-1].content if messages else ""
            )
            
            # Get LLM response
            response = self.llm.invoke(full_prompt)
            return {"messages": messages + [response]}
            
        # Add nodes to graph
        workflow.add_node("agent_step", process_step)
        
        # Add edges
        workflow.add_edge("agent_step", should_continue)
        
        return workflow

    async def analyze(self, query: str) -> Dict[str, Any]:
        """
        Execute the analysis based on the given query.
        
        Args:
            query (str): The analysis request/question
            
        Returns:
            Dict[str, Any]: Analysis results
        """
        try:
            # Add query to messages
            initial_state = {
                "messages": [HumanMessage(content=query)],
                "next_step": None,
                "tool_output": None,
                "final_answer": None
            }
            
            # Run the graph
            result = await self.agent_executor.arun(initial_state)
            
            # Extract final answer from messages
            final_message = result["messages"][-1].content if result["messages"] else ""
            
            return {
                "status": "success",
                "analysis": final_message,
                "metadata": {
                    "model": self.llm.model_name,
                    "tools_used": [tool.name for tool in self.tools]
                }
            }
        except Exception as e:
            error_msg = str(e)
            if "Missing some input keys" in error_msg:
                error_msg = f"Tool input validation error: {error_msg}"
            return {
                "status": "error",
                "error": error_msg,
                "metadata": {
                    "model": self.llm.model_name,
                    "tools_used": [tool.name for tool in self.tools]
                }
            } 