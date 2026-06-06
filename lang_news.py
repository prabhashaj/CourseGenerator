import streamlit as st
from langchain_mistralai import ChatMistralAI
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

def tavily_search(query, max_results=5):
    tavily_api_key = st.secrets.get("TAVILY_API_KEY") or os.getenv("TAVILY_API_KEY") or "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"
    if not tavily_api_key:
        raise Exception("Tavily API key not configured")
        
    payload = {
        "api_key": tavily_api_key,
        "query": query,
        "search_depth": "advanced",
        "max_results": max_results
    }
    
    import requests
    response = requests.post("https://api.tavily.com/search", json=payload, timeout=15)
    response.raise_for_status()
    res_json = response.json()
    
    results = res_json.get("results", [])
    if not results:
        raise Exception("No search results found from Tavily")
        
    formatted_results = []
    sources = []
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        url = result.get('url', 'No URL')
        content = result.get('content', 'No description')
        
        formatted_results.append(f"{i}. **{title}**\n   {content}\n   Source: {url}\n")
        sources.append(f"[{title}]({url})")
        
    search_content = "\n".join(formatted_results)
    sources_list = " | ".join(sources)
    
    return f"**Search Results (Tavily):**\n\n{search_content}\n\n**Sources for further reading:**\n{sources_list}"

def web_search_tool_with_fallback(query, max_results=5):
    try:
        print("DEBUG: Executing Tavily search...")
        return tavily_search(query, max_results)
    except Exception as e:
        print(f"DEBUG: Tavily search failed, falling back to DuckDuckGo: {e}")
        return duckduckgo_search_tool(query, max_results)

def create_langchain_agent():
    """
    Creates and returns a LangChain agent executor powered by Mistral.
    """
    # Use Mistral
    api_key = os.environ.get("MISTRAL_API_KEY") or "a9jVQQfE1QKrhhpuVTPrs78IdpL4anhW"
    
    llm = ChatMistralAI(
        model="mistral-large-latest",
        api_key=api_key,
        temperature=0.1
    ) 

    # 2. Define the Tools - simplified to just web search
    search_tool = Tool(
        name="web_search",
        func=lambda q: web_search_tool_with_fallback(q, max_results=5),
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

IMPORTANT: When you use web search, don't just repeat the search results. Instead:
1. Analyze and synthesize the information from multiple sources
2. Create a comprehensive, well-structured answer in your own words
3. Present the information in a natural, readable format
4. Include key statistics, facts, and findings
5. Provide context and explanation
6. Mention sources naturally within your response at end

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

def synthesize_search_results(query, search_results):
    """
    Use Mistral AI to synthesize search results into a comprehensive, natural answer.
    """
    try:
        # Use Mistral
        api_key = os.environ.get("MISTRAL_API_KEY") or "a9jVQQfE1QKrhhpuVTPrs78IdpL4anhW"
        
        llm = ChatMistralAI(
            model="mistral-large-latest",
            api_key=api_key,
            temperature=0.3
        )
        
        synthesis_prompt = f"""Based on the following web search results, provide a comprehensive, well-structured answer to the question: "{query}"

Web Search Results:
{search_results}

Instructions:
1. Analyze and synthesize the information from the search results
2. Create a natural, readable response in your own words
3. Include key statistics, facts, and findings where relevant
4. Provide context and explanations to make the information clear
5. Structure your answer logically with proper flow
6. Don't just copy-paste from the search results - interpret and explain
7. Keep the tone informative but conversational
8. If there are conflicting information, mention different perspectives
9. Focus on the most important and relevant information to answer the question

Write a comprehensive answer that someone could easily read and understand:"""
        
        response = llm.invoke(synthesis_prompt)
        synthesized_content = response.content if hasattr(response, 'content') else str(response)
        
        return synthesized_content
        
    except Exception as e:
        print(f"DEBUG: Synthesis failed: {e}")
        
        # Check for authentication errors
        if "401" in str(e) or "unauthorized" in str(e).lower():
            print("DEBUG: Authentication error in synthesis!")
            st.error("🔑 **Authentication Error in AI Synthesis**: Please check your Mistral API key.")
            return f"**Search Results Found** (AI synthesis failed due to authentication error):\n\n{search_results}"
        
        # Fallback to basic formatting if synthesis fails
        return f"Based on current web search results:\n\n{search_results}"

def direct_chat_fallback(query):
    """
    Direct chat with Mistral AI model as a fallback, but emphasize current information.
    """
    try:
        # Use Mistral
        api_key = os.environ.get("MISTRAL_API_KEY") or "a9jVQQfE1QKrhhpuVTPrs78IdpL4anhW"
        
        llm = ChatMistralAI(
            model="mistral-large-latest",
            api_key=api_key,
            temperature=0.0
        )
        
        # Enhanced prompt for current information
        enhanced_prompt = f"""Based on the current date being June 29, 2025, please provide the most up-to-date information about: {query}

If this is about recent events (especially 2025), prioritize the most current information available. If you don't have recent information, clearly state that web search would be needed for the latest updates.

Question: {query}"""
        
        response = llm.invoke(enhanced_prompt)
        return response.content if hasattr(response, 'content') else str(response)
    except Exception as e:
        print(f"DEBUG: Direct chat fallback failed: {e}")
        
        # Check for authentication errors
        if "401" in str(e) or "unauthorized" in str(e).lower():
            print("DEBUG: Authentication error in direct chat!")
            st.error("🔑 **Authentication Error**: Cannot access Mistral API. Please check your API key configuration.")
        
        return None

def main():
    """
    The main function for the Streamlit application.
    """
    st.title("Knowledge Hub 📰")
    st.write("Ask me anything! I'll answer directly if I know, or search the web and trusted sources if needed.")

    # Load API key from Streamlit secrets or environment
    load_dotenv()
    
    # Get Mistral API key from Streamlit secrets or environment
    mistral_api_key = None
    
    # Try Streamlit secrets first (for deployment)
    try:
        mistral_api_key = st.secrets.get("MISTRAL_API_KEY") or os.environ.get("MISTRAL_API_KEY") or "a9jVQQfE1QKrhhpuVTPrs78IdpL4anhW"
    except Exception as e:
        mistral_api_key = os.environ.get("MISTRAL_API_KEY") or "a9jVQQfE1QKrhhpuVTPrs78IdpL4anhW"
    
    # Check if we have the required API key
    if not mistral_api_key:
        st.error("🔑 **Mistral API key not found!**")
        st.stop()
    
    # Set the API key in environment
    os.environ["MISTRAL_API_KEY"] = mistral_api_key
    
    masked_key = mistral_api_key[:8] + "..." + mistral_api_key[-4:] if len(mistral_api_key) > 12 else "***"
    st.markdown(f"> ✅ **Mistral API connected** (Key: {masked_key})")

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

        # Process query
        status_placeholder.info("🚀 Processing your question...")
        
        try:
            # Try to get response from agent
            agent_response = agent_executor.invoke({"input": query})
            
            if isinstance(agent_response, dict) and "output" in agent_response:
                full_agent_output = agent_response["output"]
                
                if full_agent_output and full_agent_output.strip() != "Agent stopped due to iteration limit or time limit.":
                    status_placeholder.success("Query processed successfully!")
                    
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
                    
                    response_placeholder.markdown(complete_response)
                else:
                    # Agent output was empty, use fallback
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
                    
        except Exception as e:
            print(f"DEBUG: Error during agent execution: {e}")
            
            # Check for authentication errors
            if "401" in str(e) or "unauthorized" in str(e).lower():
                st.error("🔑 **Authentication Error**: There's an issue with the Mistral API key.")
                st.info("Please check that your API key is correctly set in Streamlit secrets and is valid.")
            else:
                st.error(f"An error occurred: {e}")
                
                # Try direct fallback
                status_placeholder.info("Trying direct chat fallback...")
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
                    status_placeholder.error("Unable to process query")

if __name__ == "__main__":
    main()
