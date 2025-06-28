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
    """Enhanced DuckDuckGo search that returns content with sources, prioritizing recent results."""
    retries = 3
    delay = 2
    
    # Add 2025 to query to prioritize recent results
    enhanced_query = f"{query} 2025" if "2025" not in query else query
    
    for attempt in range(1, retries + 1):
        try:
            with DDGS() as ddgs:
                # Search with time filter for recent results
                results = ddgs.text(enhanced_query, region='wt-wt', safesearch='off', max_results=max_results, timelimit='y')
                
                if not results:
                    # If no recent results, try without time filter
                    results = ddgs.text(query, region='wt-wt', safesearch='off', max_results=max_results)
                
                if not results:
                    return f"No search results found for: {query}"
                
                # Sort results by relevance to 2025 (prioritize recent content)
                sorted_results = []
                recent_results = []
                
                for result in results:
                    title = result.get('title', 'No title')
                    body = result.get('body', 'No description')
                    
                    # Check if result contains 2025 or recent keywords
                    if any(keyword in (title + body).lower() for keyword in ['2025', 'june 2025', 'recent', 'latest', 'breaking']):
                        recent_results.append(result)
                    else:
                        sorted_results.append(result)
                
                # Prioritize recent results first
                final_results = recent_results + sorted_results
                final_results = final_results[:max_results]  # Limit to max_results
                
                # Format results with content and sources
                formatted_results = []
                sources = []
                
                for i, result in enumerate(final_results, 1):
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
                
                return f"**Search Results (Prioritizing 2025/Recent):**\n\n{search_content}\n\n**Sources for further reading:**\n{sources_list}"
                
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
        temperature=0.1
    )
    print(f"DEBUG: LLM object created: {type(llm)}") 

    # 2. Define the Tools - simplified to just web search
    search_tool = Tool(
        name="web_search",
        func=lambda q: duckduckgo_search_tool(q, max_results=5),
        description="Search the web for current, up-to-date information. Use this when you need recent news, current events, latest developments, or real-time information."
    )
    
    # Only use web search tool to avoid confusion
    tools = [search_tool]

    # 3. Define a simpler, more reliable prompt
    prompt_template = """You are a helpful assistant that can search the web for current information when needed.

You have access to a web search tool. Use it when the question requires current, recent, or real-time information.

For questions about:
- Current events, news, breaking stories
- Recent developments (2024-2025)
- Latest products, technologies, or updates
- Real-time data (prices, weather, scores)
- When you're unsure if your knowledge is current enough

For general knowledge questions about established facts, history, science, or concepts - you can answer directly without searching.

Format your responses clearly and cite sources when you use web search.

You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
    
    prompt = PromptTemplate.from_template(prompt_template)

    # 4. Create the Agent 
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create the Agent Executor with optimized settings for better error handling
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,  # Enable verbose for debugging
        handle_parsing_errors="Check your output and make sure it conforms to the expected format! Try again.",
        max_iterations=3,  # Reduce iterations to avoid timeouts
        return_intermediate_steps=True,
        max_execution_time=30  # 30 second timeout
    )
    print(f"DEBUG: Agent executor created successfully: {type(agent_executor)}") 

    return agent_executor

def format_agent_response(agent_output):
    """
    Format the agent's response with proper brief answer and detailed information.
    """
    if not agent_output or agent_output.strip() == "":
        return "No response available", ""
    
    # Clean up the response
    clean_output = agent_output.strip()
    
    # Extract first meaningful sentence for brief answer
    sentences = clean_output.split('.')
    brief_sentence = ""
    for sentence in sentences:
        clean_sentence = sentence.strip()
        if clean_sentence and len(clean_sentence) > 30 and not clean_sentence.startswith('DEBUG'):
            brief_sentence = clean_sentence + "."
            break
    
    return brief_sentence, clean_output

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
        if action == "web_search":
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
            self.progress_bar.progress(100)  # Complete the bar
    
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
            
    except OutputParserException as e:
        print(f"DEBUG: OutputParserException occurred: {e}")
        if callback:
            callback.add_thought("Agent encountered parsing issues, switching to direct search approach...")
        
        # If parsing fails, try direct search as fallback
        try:
            search_results = duckduckgo_search_tool(query, max_results=5)
            if callback:
                callback.add_action("web_search", query)
                callback.add_observation(search_results[:300] + "...")
                callback.add_final_answer("Using search results")
            
            # Create a fake response structure to maintain consistency
            return {
                "output": f"Based on current web search results:\n\n{search_results}",
                "intermediate_steps": [
                    (type('Action', (), {'tool': 'web_search', 'tool_input': query})(), search_results)
                ]
            }
        except Exception as search_error:
            print(f"DEBUG: Fallback search also failed: {search_error}")
            return None
            
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
        
        # Check for common parsing errors
        if "output_parsing_failure" in err_msg or "could not parse" in err_msg or "invalid format" in err_msg:
            print("DEBUG: Detected parsing failure, trying direct search fallback...")
            if callback:
                callback.add_thought("Agent encountered parsing issues, switching to direct search approach...")
            
            try:
                search_results = duckduckgo_search_tool(query, max_results=5)
                if callback:
                    callback.add_action("web_search", query)
                    callback.add_observation(search_results[:300] + "...")
                    callback.add_final_answer("Using search results")
                
                # Create a fake response structure to maintain consistency
                return {
                    "output": f"Based on current web search results:\n\n{search_results}",
                    "intermediate_steps": [
                        (type('Action', (), {'tool': 'web_search', 'tool_input': query})(), search_results)
                    ]
                }
            except Exception as search_error:
                print(f"DEBUG: Fallback search also failed: {search_error}")
                return None
        
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
            
            if len(intermediate_steps) > 0:
                callback.add_thought("I need to search for current information to provide an accurate answer.")
            
            for i, step in enumerate(intermediate_steps):
                print(f"DEBUG: Step {i}: {type(step)}")
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    print(f"DEBUG: Action: {action}, Observation type: {type(observation)}")
                    
                    if hasattr(action, 'tool') and hasattr(action, 'tool_input'):
                        if action.tool == 'web_search':
                            print(f"DEBUG: Detected web search with query: {action.tool_input}")
                            callback.add_action(action.tool, action.tool_input)
                            time.sleep(2)  # Allow time for animation to show
                            # Process observation
                            callback.add_observation(str(observation)[:300] + "..." if len(str(observation)) > 300 else str(observation))
                            callback.add_thought("Now analyzing the search results to provide you with an accurate answer...")
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
    Direct chat with DeepSeek model as a fallback, but emphasize current information.
    """
    try:
        llm = ChatOpenAI(
            model="deepseek/deepseek-r1-distill-qwen-32b",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ.get("OPEN_ROUTER_KEY"),
            temperature=0.0
        )
        
        # Enhanced prompt for current information
        enhanced_prompt = f"""Based on the current date being June 26, 2025, please provide the most up-to-date information about: {query}

If this is about recent events (especially 2025), prioritize the most current information available. If you don't have recent information, clearly state that web search would be needed for the latest updates.

Question: {query}"""
        
        response = llm.invoke(enhanced_prompt)
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

        # Check if the agent actually performed a web search
        search_was_performed = False
        search_results_found = None
        
        if isinstance(raw_agent_response, dict) and "intermediate_steps" in raw_agent_response:
            intermediate_steps = raw_agent_response["intermediate_steps"]
            for step in intermediate_steps:
                if isinstance(step, tuple) and len(step) == 2:
                    action, observation = step
                    if hasattr(action, 'tool') and action.tool == 'web_search':
                        search_was_performed = True
                        search_results_found = observation
                        print(f"DEBUG: Found search results: {search_results_found[:200]}...")
                        break
        
        # Smart decision: Force search only for queries that clearly need current info
        def should_force_search(query_text):
            """Determine if a query likely needs web search based on keywords"""
            current_keywords = [
                'today', 'latest', 'recent', 'current', 'now', 'breaking', 'news', 
                'happening', 'update', '2025', '2024', 'this year', 'latest', 
                'new', 'price', 'stock', 'weather', 'live', 'real-time'
            ]
            news_keywords = ['ukraine', 'war', 'politics', 'election', 'covid', 'economy']
            tech_keywords = ['ai model', 'openai', 'chatgpt', 'latest version', 'new release']
            
            query_lower = query_text.lower()
            return any(keyword in query_lower for keyword in current_keywords + news_keywords + tech_keywords)
        
        # If agent didn't search and query seems to need current info, FORCE a search
        if not search_was_performed and should_force_search(query):
            print("DEBUG: Query needs current info but agent didn't search! Forcing web search...")
            status_placeholder.info("🔍 This query needs current information - searching the web...")
            
            # Perform direct search
            live_callback.add_thought("This question needs current information, so I'm performing a web search.")
            live_callback.add_action("web_search", query)
            
            try:
                forced_search_results = duckduckgo_search_tool(query, max_results=5)
                live_callback.add_observation(forced_search_results[:300] + "...")
                search_results_found = forced_search_results
                search_was_performed = True
                print(f"DEBUG: Forced search completed: {forced_search_results[:200]}...")
            except Exception as e:
                print(f"DEBUG: Forced search failed: {e}")
                live_callback.add_thought(f"Search failed: {e}")
        
        # Process the response based on what we have
        if search_results_found and search_was_performed:
            print("DEBUG: Processing search results")
            status_placeholder.success("Query processed successfully with current information!")
            
            # Extract sources from search results
            sources = extract_sources_from_response(search_results_found)
            
            # Create response with quick summary first
            response_text = "## 📋 Quick Answer\n\n"
            
            # Extract brief answer from search results
            search_lines = search_results_found.split('\n')
            brief_content = ""
            
            for line in search_lines:
                if line.strip() and not line.startswith('**Search Results') and not line.startswith('Source:') and '**Sources for further reading:**' not in line:
                    clean_line = line.strip()
                    if clean_line and len(clean_line) > 20:
                        brief_content = clean_line
                        break
            
            if brief_content:
                brief_answer = re.sub(r'\*\*([^*]+)\*\*', r'\1', brief_content)
                brief_answer = re.sub(r'^\d+\.\s*', '', brief_answer)
                brief_answer = brief_answer[:200] + "..." if len(brief_answer) > 200 else brief_answer
                response_text += f"**{brief_answer}**\n\n"
            else:
                response_text += "Based on current search results, here's what I found about your query.\n\n"
            
            response_text += "---\n\n"
            response_text += "**Detailed Information:**\n\n"
            
            # Extract and format all search results generically
            lines = search_results_found.split('\n')
            for line in lines:
                if line.strip() and not line.startswith('**Search Results') and not line.startswith('Source:') and '**Sources for further reading:**' not in line:
                    clean_line = line.strip()
                    if clean_line:
                        response_text += clean_line + "\n"
            response_text += "\n"
            
            # Add sources section
            if sources:
                response_text += "---\n\n### 📚 Sources:\n\n"
                for i, (title, url) in enumerate(sources, 1):
                    response_text += f"**{i}.** [{title}]({url})\n\n"
                response_text += "*💡 Click any source above to visit the original website for more information.*"
            
            response_placeholder.markdown(response_text)
            
        elif raw_agent_response and isinstance(raw_agent_response, dict) and "output" in raw_agent_response:
            # Agent provided a direct answer
            full_agent_output = raw_agent_response["output"]
            
            if full_agent_output and full_agent_output.strip() != "Agent stopped due to iteration limit or time limit.":
                print("DEBUG: Processing agent's direct answer")
                status_placeholder.success("Query processed successfully!")
                
                # Extract and display sources if any
                sources = extract_sources_from_response(full_agent_output)
                
                # Format the response properly
                brief_answer, detailed_answer = format_agent_response(full_agent_output)
                
                # Prepare the complete response with brief answer first
                complete_response = "## 📋 Quick Answer\n\n"
                
                if brief_answer:
                    complete_response += f"**{brief_answer}**\n\n"
                else:
                    complete_response += "**Here's what I found about your query.**\n\n"
                
                complete_response += "---\n\n"
                complete_response += "**Detailed Information:**\n\n"
                complete_response += detailed_answer
                
                if sources:
                    complete_response += "\n\n---\n\n### 📚 Sources:\n\n"
                    for i, (title, url) in enumerate(sources, 1):
                        complete_response += f"**{i}.** [{title}]({url})\n\n"
                    complete_response += "*💡 Click any source above to visit the original website for more information.*"
                
                response_placeholder.markdown(complete_response)
            else:
                # Agent output was empty or stopped, use fallback
                print("DEBUG: Agent output was empty, using direct chat fallback")
                status_placeholder.info("Using direct chat as fallback...")
                direct_response = direct_chat_fallback(query)
                
                if direct_response:
                    brief_answer, detailed_answer = format_agent_response(direct_response)
                    
                    formatted_response = "## 📋 Quick Answer\n\n"
                    formatted_response += f"**{brief_answer}**\n\n" if brief_answer else "**Here's what I found about your query.**\n\n"
                    formatted_response += "---\n\n**Detailed Information:**\n\n"
                    formatted_response += detailed_answer
                    
                    response_placeholder.markdown(formatted_response)
                    status_placeholder.success("Query processed successfully!")
                else:
                    response_placeholder.error("Failed to get a response from both agent and direct chat.")
                    status_placeholder.error("Unable to process query")
        else:
            # No valid response from agent, use direct chat fallback
            print("DEBUG: No valid response from agent, using direct chat fallback")
            status_placeholder.info("Using direct chat as fallback...")
            direct_response = direct_chat_fallback(query)
            
            if direct_response:
                brief_answer, detailed_answer = format_agent_response(direct_response)
                
                formatted_response = "## 📋 Quick Answer\n\n"
                formatted_response += f"**{brief_answer}**\n\n" if brief_answer else "**Here's what I found about your query.**\n\n"
                formatted_response += "---\n\n**Detailed Information:**\n\n"
                formatted_response += detailed_answer
                
                response_placeholder.markdown(formatted_response)
                status_placeholder.success("Query processed successfully!")
            else:
                response_placeholder.error("Failed to get a response from both agent and direct chat.")
                status_placeholder.error("Unable to process query")

if __name__ == "__main__":
    main()
