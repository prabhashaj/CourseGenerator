import os
import streamlit as st
from newspaper import Article
import newspaper
import nltk

# Set NLTK data path explicitly at the very beginning
nltk_data_path = os.path.join(os.path.expanduser("~"), "AppData", "Roaming", "nltk_data")
if nltk_data_path not in nltk.data.path:
    nltk.data.path.append(nltk_data_path)

# Download the 'punkt' and 'punkt_tab' tokenizers for nltk
nltk.download('punkt')
nltk.download('punkt_tab')

def get_tech_news():
    """
    Fetches and parses the latest tech news from a list of predefined global and Indian sources.
    """
    st.write("Fetching the latest tech news from worldwide sources...")

    tech_sources = [
        {'url': 'https://www.theverge.com/tech', 'brand': 'The Verge (Global)'},
        {'url': 'https://techcrunch.com/', 'brand': 'TechCrunch (Global)'},
        {'url': 'https://www.wired.com/category/technology/', 'brand': 'Wired (Global)'},
        {'url': 'https://arstechnica.com/gadgets/', 'brand': 'Ars Technica (Global)'},
        {'url': 'https://www.reuters.com/technology/', 'brand': 'Reuters Technology'},
        {'url': 'https://www.bbc.com/news/technology', 'brand': 'BBC Technology'},
        {'url': 'https://www.gadgets360.com/news', 'brand': 'Gadgets 360 (India)'},
        {'url': 'https://www.digit.in/news/technology/', 'brand': 'Digit.in (India)'}
    ]
    
    articles_data = []
    processed_titles = set()

    for source in tech_sources:
        st.write(f"Fetching from: **{source['brand']}**")
        try:
            news_source = newspaper.build(source['url'], memoize_articles=False, request_timeout=20)
            for article in news_source.articles[:3]:
                try:
                    article.download()
                    article.parse()
                    article.nlp()
                    if article.title and article.title not in processed_titles:
                        articles_data.append({
                            'title': article.title,
                            'text': article.summary,
                            'image': article.top_image,
                            'url': article.url,
                            'source': source['brand']
                        })
                        processed_titles.add(article.title)
                except Exception as e:
                    st.warning(f"Could not process an article from {source['brand']}: {e}")
        except Exception as e:
            st.error(f"Failed to build source for {source['brand']}: {e}")
            
    return articles_data

def search_news(query):
    """
    Searches for news articles based on a user's query using Google News for worldwide results.
    """
    st.write(f"Searching for '{query}' on Google News...")

    # Using Google News for search. Removed country-specific geolocation for worldwide results.
    search_url = f"https://news.google.com/search?q={query.replace(' ', '%20')}&hl=en-US&ceid=US:en"
    
    news_source = None # Initialize news_source to None
    try:
        news_source = newspaper.build(search_url, memoize_articles=False, request_timeout=20)
    except Exception as e:
        st.error(f"Failed to fetch search results from Google News: {e}")
        st.info("Please try a different search query or check back later.")
        return [] # Return empty list if building source fails

    articles_data = []
    
    # Check if news_source was successfully built before iterating
    if news_source:
        # Iterate through the top 10 search results
        for article in news_source.articles[:10]:
            try:
                article.download()
                article.parse()
                
                # Extract source domain to show the origin of the news
                source_domain = 'N/A'
                if hasattr(article, 'source_url') and article.source_url:
                    source_domain = article.source_url.split('//')[1].split('/')[0].replace('www.', '')

                articles_data.append({
                    'title': article.title,
                    'text': article.text, # Full text is often better for search results
                    'image': article.top_image,
                    'url': article.url,
                    'source': source_domain
                })
            except Exception as e:
                st.warning(f"Error processing a search result article: {e}") # Use st.warning for individual article errors
                
    return articles_data

def trending_tech_topics():
    """
    Fetches trending technology topics from Google Trends (using pytrends) and displays them.
    """
    import pandas as pd
    from pytrends.request import TrendReq
    st.write("Fetching trending technology topics...")
    try:
        pytrends = TrendReq(hl='en-US', tz=360)
        kw_list = ['technology', 'AI', 'blockchain', 'cybersecurity', 'cloud computing', 'robotics', 'semiconductors', 'quantum computing']
        pytrends.build_payload(kw_list, cat=0, timeframe='now 7-d', geo='', gprop='')
        trends = pytrends.related_queries()
        st.subheader("Trending Tech Topics (Google Trends)")
        for kw in kw_list:
            if kw in trends and trends[kw]['top'] is not None:
                st.markdown(f"**{kw.title()}**")
                df = trends[kw]['top']
                for i, row in df.head(5).iterrows():
                    st.write(f"- {row['query']} (score: {row['value']})")
            else:
                st.write(f"No trending data for {kw}.")
        st.info("Trends are based on Google search data from the past week.")
    except Exception as e:
        st.error(f"Failed to fetch trending topics from Google Trends: {e}")
        st.info("This might be a temporary issue or related to the request parameters. Please try again later.")

def main():
    """
    The main function for the Streamlit application.
    """
    # Remove sidebar navigation and main app title/description
    # st.sidebar.title("🌍 Tech News Navigation")
    # page = st.sidebar.radio(
    #     "Go to",
    #     ("Latest Tech News", "Trending Tech Topics", "Search Tech News")
    # )

    # st.title("🌍 Worldwide Tech News Retriever")
    # st.write("Your one-stop solution for the latest in technology news from around the globe.")

    # Instead, just show the latest tech news directly
    news_data = get_tech_news()
    if news_data:
        for article in news_data:
            image_url = article['image'] if article['image'] else "https://placehold.co/600x400/262626/FFFFFF?text=No+Image"
            col1, col2 = st.columns([1, 3])
            with col1:
                st.image(image_url, use_container_width=True)
            with col2:
                st.subheader(article['title'])
                st.caption(f"Source: **{article.get('source', 'N/A')}**")
                summary = article['text']
                st.write(summary[:300] + '...' if summary and len(summary) > 300 else summary or 'No description available.')
                st.markdown(f"[Read more]({article['url']})", unsafe_allow_html=True)
            st.markdown("---")
    else:
        st.info("No news articles found. Please try a different search or check back later.")

    # Optionally, you can add a button to show trending topics or search, but not in the sidebar
    # If you want to keep those features, you can add them as buttons or sections here.

if __name__ == "__main__":
    main()
