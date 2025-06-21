import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_react_agent, Tool
from langchain_community.agent_toolkits.load_tools import load_tools 
from langchain.prompts import PromptTemplate
import os
import re
from dotenv import load_dotenv
import time
from duckduckgo_search import DDGS

def duckduckgo_search_tool(query, max_results=5):
    """Unlimited DuckDuckGo search using DDGS, with rate limit handling."""
    retries = 3
    delay = 2
    for attempt in range(1, retries + 1):
        try:
            with DDGS() as ddgs:
                results = ddgs.text(query, region='wt-wt', safesearch='off', max_results=max_results)
                # Return URLs from search results
                return '\n'.join([r['href'] for r in results if 'href' in r])
        except Exception as e:
            if attempt < retries:
                time.sleep(delay)
                delay *= 2
            else:
                return f"[DuckDuckGo rate limit hit. Please try again later. Error: {e}]"

def create_langchain_agent():
    """
    Creates and returns a LangChain agent executor powered by Google Gemini.
    """
    # 1. Define the LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.0,
        convert_system_message_to_human=True,
        streaming=True
    )
    print(f"DEBUG: LLM object created: {type(llm)}") 

    # 2. Define the Tools
    search_tool = Tool(
        name="duckduckgo_search",
        func=lambda q: duckduckgo_search_tool(q, max_results=5),
        description="Searches the web for up-to-date information using DuckDuckGo. Returns a list of URLs."
    )
    
    # Tools list includes only search_tool
    tools = [search_tool]

    # 3. Define the Prompt
    # Modified prompt to explicitly ask for sources in Markdown link format within the Final Answer
    prompt_template = """
You are an expert knowledge assistant. Your goal is to answer the user's query as accurately as possible.

- If you are certain you know the answer (from your own knowledge), answer directly and do NOT make up information.
- If you are NOT certain, or if the query requires up-to-date information, use the available tool to find the answer.
- Never hallucinate or guess. If you don't know, use the tool.

You have access to the following tool: {tools}
- Use the search tool for up-to-date information.

To use a tool, please use the following format:
```
Thought: Do I need to use a tool? Yes
Action: The action to take. Should be one of [{tool_names}]
Action Input: The input to the action
Observation: The result of the action
```

When you have a comprehensive answer based on your knowledge ONLY, use the format:
```
Thought: Do I need to use a tool? No
Final Answer: [Your comprehensive answer here]
```

When you have a comprehensive answer that required using a tool, you MUST use the format:
```
Thought: Do I need to use a tool? Yes
Final Answer: [Your comprehensive, well-cited answer here.
Sources:
[Source Title 1](URL 1)
[Source Title 2](URL 2)
...]
```

Begin!

Question: {input}
{agent_scratchpad}
    """
    prompt = PromptTemplate.from_template(prompt_template)

    # 4. Create the Agent
    agent = create_react_agent(llm, tools, prompt)

    # 5. Create the Agent Executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5,
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

def invoke_agent_safely(agent_executor, query):
    """
    Invokes the agent executor with robust error handling for common issues.
    Returns the agent's raw response or None if an unrecoverable error occurs.
    """
    try:
        raw_response = agent_executor.invoke({"input": query})
        return raw_response
    except AttributeError as e:
        if "'NoneType' object has no attribute 'get'" in str(e):
            print(f"CRITICAL ERROR: Agent invocation returned None, leading to AttributeError: {e}")
            st.error("A critical internal error occurred: The agent's response was unexpectedly empty. This might indicate an issue with the underlying LLM service or a dependency.")
            return None
        else:
            print(f"DEBUG: Unexpected AttributeError during invoke: {e}")
            st.error(f"An unexpected error occurred during agent execution: {e}. Please try again later.")
            return None
    except Exception as e:
        err_msg = str(e).lower()
        print(f"DEBUG: General exception during agent invocation: {e}")
        if "rate limit" in err_msg or "quota" in err_msg:
            st.error("Sorry, the API service is temporarily rate-limited. Please try again in a few minutes.")
        else:
            st.error(f"An unexpected error occurred during agent execution: {e}. Please try again later.")
        return None

def main():
    """
    The main function for the Streamlit application.
    """
    st.title("Knowledge Hub 📰")
    st.write("Ask me anything! I'll answer directly if I know, or search the web and trusted sources if needed.")

    # Load API key from .env or secrets.toml
    load_dotenv()
    google_api_key = (
        os.getenv("GOOGLE_API_KEY")
        or st.secrets.get("GOOGLE_API_KEY", "")
    )
    if not google_api_key:
        st.error("Google API key not found. Please set GOOGLE_API_KEY in your .env file or Streamlit secrets.")
        st.stop()
    os.environ["GOOGLE_API_KEY"] = google_api_key

    # Main input field and button stacked vertically
    query = st.text_input("Enter your query:", placeholder="e.g., What is Google's latest AI model?", key="news_query")
    get_news = st.button("Get Answer", key="get_news_btn")

    # Placeholders for dynamic content
    agent_steps_placeholder = st.empty()
    response_placeholder = st.empty()

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

        # Display an initial status message for the user
        status_message = st.empty()
        status_message.info("Agent is thinking and searching...")

        with st.spinner("Processing your query..."):
            raw_agent_response = invoke_agent_safely(agent_executor, query)

        # Debugging: Print the type and content of raw_agent_response
        print(f"DEBUG: Type of raw_agent_response after invoke_agent_safely: {type(raw_agent_response)}")
        print(f"DEBUG: Raw agent response after invoke_agent_safely: {raw_agent_response}")

        if raw_agent_response is None:
            # If invoke_agent_safely returned None, an error message has already been shown.
            status_message.error("Failed to get a response from the agent.")
            st.stop()
        elif isinstance(raw_agent_response, dict) and "output" in raw_agent_response:
            full_agent_output = raw_agent_response["output"]
            
            # Update status message to indicate completion
            status_message.success("Query processed successfully!")

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

            # Display agent's reasoning steps
            with agent_steps_placeholder.expander("Agent's Actions & Reasoning", expanded=True):
                if steps:
                    for i, step in enumerate(steps, 1):
                        st.markdown(f"**Step {i}:**")
                        st.markdown(f"- **Thought:** {step['thought']}")
                        st.markdown(f"- **Action:** {step['action']}")
                        st.markdown(f"- **Action Input:** {step['input']}")
                        st.markdown(f"- **Observation:** {step['observation']}")
                        st.markdown("---")
                else:
                    # Inform user that no intermediate steps were found. This is expected if LLM answers directly.
                    st.info("No intermediate steps found in the response (agent might have answered directly).")

            # Show the main final answer
            if final_answer_for_display.strip(): # Check if the extracted answer text is not empty
                response_placeholder.markdown(final_answer_for_display)
            else:
                # If no direct answer from LLM, indicate it and suggest checking sources/reasoning
                response_placeholder.info("The agent did not provide a direct answer from its knowledge for this query. Please refer to the 'Agent's Actions & Reasoning' for details if any tool was used, or refer to sources below.")
        else:
            final_display_output = f"The agent returned an unexpected response format: {raw_agent_response}. Please try again or refine your query."
            status_message.error(final_display_output)
            st.stop() # Stop if format is unexpected, as further processing will likely fail

if __name__ == "__main__":
    main()