import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.agent_toolkits.load_tools import load_tools 
from langchain.prompts import PromptTemplate
import os
import re
from dotenv import load_dotenv
import time
import threading
from duckduckgo_search import DDGS
from langchain.schema import OutputParserException, AgentAction, AgentFinish
from langchain.agents.agent import AgentOutputParser

def duckduckgo_search_tool(query, max_results=5):
    """Enhanced DuckDuckGo search that returns content with sources."""
    retries = 3
    delay = 2
    for attempt in range(1, retries + 1):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, region='wt-wt', safesearch='off', max_results=max_results)
                
                if not results:
                    return f"No search results found for: {query}"
                
                # Format results with content and sources
                formatted_results = []
                sources = []
                
                for i, result in enumerate(results, 1):
                    title = result.get('title', 'No title')
                    href = result.get('href', 'No URL')
                    body = result.get('body', 'No description')
                    
                    # Truncate body to reasonable length
                    if len(body) > 200:
                        body = body[:200] + "..."
                    
                    formatted_results.append(f"{i}. **{title}**\n   {body}\n   Source: {href}\n")
                    sources.append(f"[{title}]({href})")
                
                search_content = "\n".join(formatted_results)
                sources_list = " | ".join(sources)
                
                return f"**Search Results:**\n\n{search_content}\n\n**Sources for further reading:**\n{sources_list}"
                
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
            else:
                return f"[DuckDuckGo search failed after {retries} attempts. Error: {e}]"

def create_langchain_agent():
    """
    Creates and returns a LangChain agent executor powered by DeepSeek via OpenRouter.
    """
    # 1. Define the LLM using OpenRouter
    llm = ChatOpenAI(
        model="deepseek/deepseek-r1-distill-qwen-32b",
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ.get("OPEN_ROUTER_KEY"),
        temperature=0.0
    )
    print(f"DEBUG: LLM object created: {type(llm)}") 

    # 2. Define the Tools
    search_tool = Tool(
        name="duckduckgo_search",
        func=lambda q: duckduckgo_search_tool(q, max_results=5),
        description="Searches the web for current, up-to-date information. Use this for recent events, current news, latest developments, or when you need the most recent information about a topic. Returns formatted results with sources."
    )
    
    # Tools list includes only search_tool
    tools = [search_tool]

    # 3. Define the Prompt with stricter format requirements
    prompt_template = """You are a research assistant. You MUST use the exact format below.

RULE: For ANY question about recent events, news, current data, or anything that might need up-to-date information, you MUST search the web first.

REQUIRED FORMAT:

Step 1 - Decide if you need web search:
Thought: [Analyze if this needs current information]

Step 2A - If you need web search (USE THIS FOR MOST QUESTIONS):
Action: duckduckgo_search
Action Input: [your search query]
Observation: [results will appear]
Thought: Based on search results, I can answer.
Final Answer: [answer with sources]

Step 2B - If you DON'T need web search (only for basic definitions):
Final Answer: [answer from knowledge]

IMPORTANT: Always start with "Thought:" and use web search for anything that could be current.

Tools: {tools}
Tool names: {tool_names}

Question: {input}
{agent_scratchpad}"""
    prompt = PromptTemplate.from_template(prompt_template)

    # 4. Create the Agent 
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create the Agent Executor with optimized settings
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Enable verbose for debugging
        handle_parsing_errors=True,
        max_iterations=5,  # Increase iterations
        return_intermediate_steps=True
    )
    print(f"DEBUG: Agent executor created successfully: {type(agent_executor)}") 

    return agent_executor

def parse_agent_steps(agent_output):
    """
    Parses the agent's output into steps for display.
    Separates the main answer text from the sources.
    """
    if agent_output is None:
        print("DEBUG: parse_agent_steps received None as agent_output. Returning empty.")
        return [], "", [] # Return empty steps, answer, and sources

    steps = []
    # Regex to capture Thought, Action, Action Input, and Observation
    action_pattern = r"Thought:(.*?)\n(?:Action:(.*?)\nAction Input:(.*?)\nObservation:(.*?)(?=\nThought:|\Z))"
    matches = re.findall(action_pattern, agent_output, re.DOTALL)
    for match in matches:
        steps.append({
            "thought": match[0].strip(),
            "action": match[1].strip(),
            "input": match[2].strip(),
            "observation": match[3].strip()
        })
    
    main_answer_text = ""
    extracted_links = []

    # Find the "Final Answer:" header
    final_answer_header_match = re.search(r"Final Answer:\s*", agent_output, re.DOTALL)

    if final_answer_header_match:
        # Get all content after "Final Answer:"
        content_after_final_answer = agent_output[final_answer_header_match.end():].strip()

        # Try to find "Sources:" within this content
        sources_header_match = re.search(r"\nSources:\s*", content_after_final_answer, re.DOTALL)

        if sources_header_match:
            # Content before "Sources:" is the main answer
            main_answer_text = content_after_final_answer[:sources_header_match.start()].strip()
            # Content after "Sources:" is the potential sources block
            sources_block = content_after_final_answer[sources_header_match.end():].strip()
            
            # Extract links from the sources_block
            extracted_links = re.findall(r'\[([^\]]+)\]\((https?:\/\/[^)]+)\)', sources_block)
        else:
            # No "Sources:" header found, so the entire content after "Final Answer:" is the main answer.
            main_answer_text = content_after_final_answer
            # Still attempt to extract links from this main answer text, in case they are embedded without a "Sources:" header.
            extracted_links = re.findall(r'\[([^\]]+)\]\((https?:\/\/[^)]+)\)', main_answer_text)
    else:
        # If "Final Answer:" header itself is not found, treat the whole output as the answer (fallback).
        # This might happen if the agent's output format deviates significantly.
        main_answer_text = agent_output
        extracted_links = re.findall(r'\[([^\]]+)\]\((https?:\/\/[^)]+)\)', main_answer_text)

    return steps, main_answer_text, extracted_links

def extract_sources_from_response(agent_output):
    """Extract sources from the agent's response for display"""
    sources = []
    
    print(f"DEBUG: Extracting sources from: {agent_output[:200]}...")
    
    # Look for markdown links in the format [title](url) throughout the text
    markdown_links = re.findall(r'\[([^\]]+)\]\((https?://[^\s)]+)\)', agent_output)
    print(f"DEBUG: Found {len(markdown_links)} markdown links")
    sources.extend(markdown_links)
    
    # Look for "Sources for further reading:" section
    sources_match = re.search(r'Sources for further reading:\s*(.+)', agent_output, re.DOTALL | re.IGNORECASE)
    if sources_match:
        print("DEBUG: Found 'Sources for further reading' section")
        sources_text = sources_match.group(1)
        additional_links = re.findall(r'\[([^\]]+)\]\((https?://[^\s)]+)\)', sources_text)
        sources.extend(additional_links)
    
    # Look for "Source:" patterns (from search results)
    source_lines = re.findall(r'Source:\s*(https?://[^\s]+)', agent_output)
    print(f"DEBUG: Found {len(source_lines)} source lines")
    for i, url in enumerate(source_lines):
        # Extract domain name as title if no proper title found
        domain = re.search(r'https?://(?:www\.)?([^/]+)', url)
        title = domain.group(1) if domain else f"Source {i+1}"
        sources.append((title, url))
    
    # Remove duplicates while preserving order
    unique_sources = []
    seen_urls = set()
    for title, url in sources:
        if url not in seen_urls:
            unique_sources.append((title, url))
            seen_urls.add(url)
    
    print(f"DEBUG: Final unique sources: {len(unique_sources)}")
    return unique_sources

class LiveAgentCallback:
    """Enhanced callback handler with web search animation and reasoning display"""
    def __init__(self, steps_placeholder, status_placeholder):
        self.steps_placeholder = steps_placeholder
        self.status_placeholder = status_placeholder
        self.is_searching = False
        self.progress_bar = None
        self.reasoning_steps = []
        
    def add_thought(self, thought):
        """Show thinking status and capture reasoning"""
        self.status_placeholder.info("🤔 Thinking...")
        self.reasoning_steps.append(f"🤔 **Thought:** {thought}")
        self._update_reasoning_display()
    
    def add_action(self, action, action_input):
        """Show action status with web search animation"""
        if action == "duckduckgo_search":
            self.is_searching = True
            self.status_placeholder.info(f"🌐 Searching the web: **{action_input}**")
            self.reasoning_steps.append(f"🌐 **Action:** Searching the web for '{action_input}'")
            self._update_reasoning_display()
            
            # Create and run progress bar animation
            self.progress_bar = st.progress(0)
            self._animate_progress()
                
        else:
            self.status_placeholder.info(f"⚡ Processing...")
            self.reasoning_steps.append(f"⚡ **Action:** {action}")
            self._update_reasoning_display()
    
    def _animate_progress(self):
        """Animate the progress bar"""
        if self.progress_bar:
            for i in range(1, 101):
                if not self.is_searching:  # Stop if search is done
                    break
                self.progress_bar.progress(i)
                time.sleep(0.02)  # Smooth animation
    
    def add_observation(self, observation):
        """Show observation and stop animation"""
        self.is_searching = False  # Stop animation
        if self.progress_bar:
            self.progress_bar.empty()  # Remove progress bar
        self.status_placeholder.success("📊 Found web results, analyzing...")
        
        # Add truncated observation to reasoning
        obs_preview = observation[:200] + "..." if len(observation) > 200 else observation
        self.reasoning_steps.append(f"📊 **Found Results:** {obs_preview}")
        self._update_reasoning_display()
    
    def add_final_answer(self, answer):
        """Show completion status"""
        self.is_searching = False  # Ensure animation stops
        if self.progress_bar:
            self.progress_bar.empty()  # Clean up progress bar
        self.status_placeholder.success("✅ Answer ready!")
        self.reasoning_steps.append("✅ **Completed:** Answer ready!")
        self._update_reasoning_display()
    
    def _update_reasoning_display(self):
        """Update the reasoning display in the UI"""
        if self.steps_placeholder:
            reasoning_text = "\n\n".join(self.reasoning_steps)
            self.steps_placeholder.markdown(reasoning_text)

def invoke_agent_safely(agent_executor, query, callback=None):
    """
    Invokes the agent executor with live updates and robust error handling.
    """
    try:
        # If we have a callback, we'll monitor the agent's progress
        if callback:
            # Create a custom invoke that monitors intermediate steps
            return invoke_agent_with_live_updates(agent_executor, query, callback)
        else:
            # Standard invoke
            raw_response = agent_executor.invoke({"input": query})
            return raw_response
            
    except AttributeError as e:
        if "'NoneType' object has no attribute 'get'" in str(e):
            print(f"CRITICAL ERROR: Agent invocation returned None, leading to AttributeError: {e}")
            st.error("A critical internal error occurred: The agent's response was unexpectedly empty.")
            return None
        else:
            print(f"DEBUG: Unexpected AttributeError during invoke: {e}")
            st.error(f"An unexpected error occurred during agent execution: {e}")
            return None
    except Exception as e:
        err_msg = str(e).lower()
        print(f"DEBUG: General exception during agent invocation: {e}")
        if "rate limit" in err_msg or "quota" in err_msg:
            st.error("Sorry, the API service is temporarily rate-limited. Please try again in a few minutes.")
        else:
            st.error(f"An unexpected error occurred during agent execution: {e}")
        return None

def invoke_agent_with_live_updates(agent_executor, query, callback):
    """
    Agent invocation with proper live UI updates and reasoning display
    """
    try:
        # Show initial thinking status
        callback.add_thought("Analyzing your question and deciding on the best approach...")
        
        # Execute agent and capture intermediate steps
        agent_response = agent_executor.invoke({"input": query})
        
        # Process intermediate steps to detect and display actions
        if isinstance(agent_response, dict) and "intermediate_steps" in agent_response:
            intermediate_steps = agent_response["intermediate_steps"]
            print(f"DEBUG: Found {len(intermediate_steps)} intermediate steps")
            
            for i, step in enumerate(intermediate_steps):
                print(f"DEBUG: Step {i}: {type(step)}")
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    print(f"DEBUG: Action: {action}, Observation type: {type(observation)}")
                    
                    if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                        if action.tool == 'duckduckgo_search':
                            print(f"DEBUG: Detected web search with query: {action.tool_input}")
                            callback.add_action(action.tool, action.tool_input)
                            time.sleep(3)  # Allow time for animation
                            callback.add_observation(str(observation)[:200] + "...")
                            callback.add_thought("Now I'll analyze these search results to provide you with an accurate answer.")
        else:
            # If no intermediate steps, it means the agent answered directly from knowledge
            callback.add_thought("I can answer this from my existing knowledge without needing to search.")
        
        # Show completion
        if isinstance(agent_response, dict) and "output" in agent_response:
            output = agent_response["output"]
            if output and output != "Agent stopped due to iteration limit or time limit.":
                callback.add_final_answer(output)
        
        return agent_response
        
    except Exception as e:
        print(f"DEBUG: Error in live agent invocation: {e}")
        # Fallback to direct execution
        callback.add_thought("Switching to fallback processing method...")
        return agent_executor.invoke({"input": query})

def direct_chat_fallback(query):
    """
    Direct chat with DeepSeek model as a fallback when agent parsing fails.
    """
    try:
        llm = ChatOpenAI(
            model="deepseek/deepseek-r1-distill-qwen-32b",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPEN_ROUTER_KEY"),
            temperature=0.0
        )
        
        response = llm.invoke(f"Please provide a comprehensive answer to this question: {query}")
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"DEBUG: Direct chat fallback failed: {e}")
        return None

def main():
    """
    The main function for the Streamlit application.
    """
    st.title("Knowledge Hub 📰")
    st.write("Ask me anything! I'll answer directly if I know, or search the web and trusted sources if needed.")

    # Load API key from Streamlit secrets or environment variables
    load_dotenv()
    
    try:
        # Try to get from Streamlit secrets first (for deployment)
        openrouter_api_key = st.secrets.get("OPEN_ROUTER_KEY")
    except:
        # Fall back to environment variables (for local development)
        openrouter_api_key = os.getenv("OPEN_ROUTER_KEY")
    
    if not openrouter_api_key:
        st.error("🔑 OpenRouter API key not found. Please set OPEN_ROUTER_KEY in Streamlit secrets for deployment, or in environment variables for local development.")
        st.info("💡 **For Streamlit Cloud:** Add your API key in the app settings under 'Secrets management'")
        st.stop()
    os.environ["OPEN_ROUTER_KEY"] = openrouter_api_key

    # Main input field and button stacked vertically
    query = st.text_input("Enter your query:", placeholder="e.g., What is Google's latest AI model?", key="news_query")
    get_news = st.button("Get Answer", key="get_news_btn")

    # Placeholders for dynamic content
    response_placeholder = st.empty()
    status_placeholder = st.empty()

    if get_news:
        if not query:
            st.warning("Please enter a query.")
            st.stop()

        try:
            agent_executor = create_langchain_agent()
        except Exception as e:
            st.error(f"Failed to initialize the agent: {e}. Please check your environment and dependencies.")
            st.stop()

        raw_agent_response = None
        final_answer_for_display = "" # Will hold the main answer text
        extracted_links = [] # Will hold parsed links

        # Display initial status
        status_placeholder.info("🚀 Getting ready to answer your question...")
        
        # Create an expandable section for agent reasoning
        with st.expander("🤖 Agent's Actions & Reasoning", expanded=True):
            steps_placeholder = st.empty()

        # Create live callback for real-time updates
        live_callback = LiveAgentCallback(steps_placeholder, status_placeholder)

        # Process query with live updates
        raw_agent_response = invoke_agent_safely(agent_executor, query, live_callback)

        # Debugging: Print the type and content of raw_agent_response
        print(f"DEBUG: Type of raw_agent_response after invoke_agent_safely: {type(raw_agent_response)}")
        print(f"DEBUG: Raw agent response after invoke_agent_safely: {raw_agent_response}")

        if raw_agent_response is None:
            # If invoke_agent_safely returned None, try direct chat as fallback
            status_placeholder.info("Trying direct chat as fallback...")
            direct_response = direct_chat_fallback(query)
            if direct_response:
                status_placeholder.success("Query processed successfully!")
                response_placeholder.markdown(direct_response)
            else:
                status_placeholder.error("Failed to get a response from the agent.")
            st.stop()
        elif isinstance(raw_agent_response, dict) and "output" in raw_agent_response:
            full_agent_output = raw_agent_response["output"]
            
            # Check if agent still returned the iteration limit message
            if full_agent_output == "Agent stopped due to iteration limit or time limit.":
                status_placeholder.info("Agent hit parsing limits, trying direct chat...")
                direct_response = direct_chat_fallback(query)
                if direct_response:
                    status_placeholder.success("Query processed successfully!")
                    response_placeholder.markdown(direct_response)
                else:
                    status_placeholder.error("Failed to get a response from both agent and direct chat.")
                st.stop()
            
            # Update status message to indicate completion
            status_placeholder.success("Query processed successfully!")

            # Parse steps and the main final answer text
            steps, final_answer_for_display, extracted_links = parse_agent_steps(full_agent_output)
            
            # Debugging: Print parsed results
            print(f"DEBUG: main_final_answer_text (from parser): {repr(final_answer_for_display)}")
            print(f"DEBUG: Extracted links (from parser): {extracted_links}")

            # Deduplicate extracted links
            unique_links = []
            seen_urls = set()
            for title, url in extracted_links:
                if url not in seen_urls:
                    unique_links.append((title, url))
                    seen_urls.add(url)
            extracted_links = unique_links

            # Show the main final answer in response placeholder
            if final_answer_for_display.strip():
                # Extract and display sources if any
                sources = extract_sources_from_response(full_agent_output)
                
                # Clean up the main answer text (remove source URLs that appear inline)
                clean_answer = re.sub(r'Source:\s*https?://[^\s]+', '', final_answer_for_display)
                clean_answer = re.sub(r'\n\s*\n', '\n\n', clean_answer)  # Clean up extra newlines
                
                # Prepare the complete response with sources
                complete_response = clean_answer.strip()
                
                if sources:
                    complete_response += "\n\n---\n\n### 📚 Sources:\n\n"
                    for i, (title, url) in enumerate(sources, 1):
                        # Create clickable numbered links
                        complete_response += f"**{i}.** [{title}]({url})\n\n"
                    complete_response += "*💡 Click any source above to visit the original website for more information.*"
                
                # Display the complete response with sources
                response_placeholder.markdown(complete_response)
                        
            else:
                # If no final answer was parsed, show the raw output but try to extract sources
                sources = extract_sources_from_response(full_agent_output)
                display_text = full_agent_output
                
                if sources:
                    display_text += "\n\n---\n\n### 📚 Sources:\n\n"
                    for i, (title, url) in enumerate(sources, 1):
                        display_text += f"**{i}.** [{title}]({url})\n\n"
                    display_text += "*💡 Click any source above to visit the original website for more information.*"
                
                response_placeholder.markdown(display_text)
        else:
            final_display_output = f"The agent returned an unexpected response format: {raw_agent_response}. Please try again or refine your query."
            status_placeholder.error(final_display_output)
            st.stop() # Stop if format is unexpected, as further processing will likely fail

        # Fallback: If parsing failed and no valid answer was found, use direct chat as a last resort
        if not final_answer_for_display.strip():
            st.warning("The agent's response could not be parsed into a valid answer. Trying direct chat fallback...")
            with st.spinner("Chatting with the AI..."):
                fallback_response = direct_chat_fallback(query)
            
            if fallback_response:
                response_placeholder.markdown(fallback_response)
            else:
                response_placeholder.error("Direct chat fallback also failed. Please try again later.")

if __name__ == "__main__":
    main()
