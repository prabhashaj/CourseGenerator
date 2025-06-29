#  VidhyAI - Course Generator

A comprehensive AI-powered learning platform that creates custom courses, enables document-based learning, provides intelligent chat assistance, and offers interactive study tools.

## ✨ Features

###  Course Generator & Tracker
- **AI-Generated Courses**: Create custom course outlines on any topic using Google's Gemini AI
- **Progress Tracking**: Monitor your learning progress with completion tracking for each chapter
- **Detailed Content Generation**: Get comprehensive chapter content with examples, explanations, and summaries
- **Interactive Quizzes**: Test your knowledge with AI-generated quizzes for each module
- **Customizable Difficulty**: Choose from Beginner, Intermediate, or Advanced levels
- **Flexible Module Structure**: Configure number of modules, chapters per module, and reading time

### 📄 Document-Based Course Creation
- **Upload & Learn**: Transform your PDF and TXT documents into structured courses
- **Intelligent Content Extraction**: AI analyzes your documents and creates organized learning modules
- **Document Chat**: Ask questions about your uploaded documents with context-aware responses
- **Course Structure**: Generate courses from your own materials with customizable difficulty levels

### 🤖 RAG Chatbot (Chat with Documents)
- **Document Intelligence**: Upload multiple PDFs/TXT files and chat with their content
- **Advanced Search**: Uses FAISS vector search with Maximum Marginal Relevance (MMR)
- **Source Attribution**: Get answers with proper source citations
- **Document Summaries**: Automatic document summarization for quick overviews
- **Conversational Memory**: Maintains context across multiple queries

### 🎮 PlayZone - Interactive Learning
- **AI Flashcards**: Generate custom flashcards on any topic
- **Difficulty Levels**: Cards categorized as Easy, Medium, or Hard
- **Score Tracking**: Keep track of correct and incorrect answers
- **Interactive Interface**: Flip cards, mark answers, and track progress
- **Beautiful UI**: Enhanced visual design with dark mode support

### 📰 Knowledge Hub
- **Real-time Information**: Get current news and information using web search
- **AI-Powered Research**: Uses DeepSeek AI model via OpenRouter for intelligent responses
- **Source Verification**: Provides sources for all web-searched information
- **Current Events**: Stay updated with the latest developments in any field

## 🛠️ Technology Stack

- **Frontend**: Streamlit
- **AI Models**: 
  - Google Gemini 2.0 Flash (Primary)
  - DeepSeek R1 Distill (Knowledge Hub)
- **Vector Database**: FAISS for document embeddings
- **Document Processing**: PyPDF2, LangChain document loaders
- **Web Search**: DuckDuckGo Search API
- **Language Framework**: LangChain for AI orchestration

## 📋 Prerequisites

- Python 3.8 or higher
- Google Gemini API key
- OpenRouter API key (for Knowledge Hub feature)

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone <https://github.com/prabhashaj/CourseGenerator.git>
cd CourseGenerator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Set Up Environment Variables

Create a `.env` file in the project root:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GOOGLE_API_KEY=your_gemini_api_key_here
OPEN_ROUTER_KEY=your_openrouter_api_key_here
```

### 4. Run the Application
```bash
streamlit run appx.py
```

## ☁️ Deployment (Streamlit Cloud)

### 1. Push to GitHub
Ensure your code is in a GitHub repository.

### 2. Configure Secrets
In Streamlit Cloud, add these secrets:
```toml
GEMINI_API_KEY = "your_gemini_api_key_here"
GOOGLE_API_KEY = "your_gemini_api_key_here"  
OPEN_ROUTER_KEY = "your_openrouter_api_key_here"
```

### 3. Deploy
- Connect your GitHub repository to Streamlit Cloud
- Set main file as `appx.py`
- Deploy!

## 📁 Project Structure

```
CourseGenerator/
├── appx.py                     # Main application entry point
├── course_generator.py         # Core course generation functionality
├── document_course_creator.py  # Document-based course creation
├── rag_chatbot.py             # RAG chatbot implementation
├── play_zone.py               # Interactive flashcards and games
├── lang_news.py               # Knowledge hub with web search
├── quiz_utils.py              # Quiz generation utilities
├── requirements.txt           # Python dependencies
├── Procfile                   # Deployment configuration
├── .env                       # Environment variables (local)
├── faiss_index/              # Vector database storage
├── faiss_store/              # Additional vector storage
├── output/                   # Generated course files
├── temp_course_docs/         # Temporary document storage
├── temp_docs/                # Temporary file processing
└── tests/                    # Test files
```

## 🎯 How to Use

### Creating Your First Course

1. **Select Mode**: Choose "Course Generator & Quizzes" from the sidebar
2. **Enter Topic**: Type your desired course topic (e.g., "Machine Learning")
3. **Configure Settings**: 
   - Choose difficulty level
   - Set number of modules
   - Select chapters per module
   - Set reading time per module
4. **Generate**: Click "Generate New Course Outline"
5. **Explore**: Browse modules, chapters, and generate detailed content
6. **Test Knowledge**: Take quizzes for each module

### Document-Based Learning

1. **Upload Documents**: Choose "Course Creation from Documents"
2. **Add Files**: Upload PDF or TXT files
3. **Configure**: Set difficulty and module preferences
4. **Generate Course**: AI will analyze and structure your documents
5. **Learn**: Navigate through auto-generated modules and chapters

### Interactive Study

1. **Access PlayZone**: Select "🎮 PlayZone" from sidebar
2. **Choose Topic**: Enter any subject for flashcard generation
3. **Set Quantity**: Choose number of flashcards (1-20)
4. **Study**: Flip cards, test yourself, and track your score

### Research & Current Info

1. **Knowledge Hub**: Select "Knowledge Hub 📰"
2. **Ask Questions**: Enter queries about current events, tech, or any topic
3. **Get Answers**: Receive AI-powered responses with web sources
4. **Verify Sources**: Check provided links for additional information

## ⚙️ Configuration

### AI Model Settings
Adjust these parameters in the sidebar:
- **Temperature**: Controls creativity (0.0 - 1.0)
- **Max Tokens**: Maximum response length
- **Top K**: Limits vocabulary for responses
- **Top P**: Controls diversity of responses

### Course Customization
- **Difficulty Levels**: Beginner, Intermediate, Advanced
- **Module Count**: 1-12 modules per course
- **Chapter Count**: 2-8 chapters per module
- **Reading Time**: 2, 5, or 10 minutes per module

## 🔧 API Keys Setup

### Google Gemini API
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Add to your environment variables

### OpenRouter API (Optional - for Knowledge Hub)
1. Visit [OpenRouter](https://openrouter.ai/)
2. Sign up and get API key
3. Add to your environment variables

## 🎨 Features in Detail

### Smart Content Generation
- **Adaptive Length**: Content adapts based on course size
- **Completion Checking**: Ensures all content is properly finished
- **Retry Logic**: Automatic retries for incomplete content
- **Token Optimization**: Efficient use of AI model limits

### Advanced RAG System
- **Semantic Search**: Find relevant information across documents
- **Source Attribution**: Every answer includes source references
- **Memory Management**: Maintains conversation context
- **Multi-document Support**: Process multiple files simultaneously

### Interactive Learning Tools
- **Progress Tracking**: Visual progress bars and completion percentages
- **Quiz Generation**: AI creates relevant questions from content
- **Flashcard System**: Adaptive difficulty and spaced repetition
- **Score Analytics**: Track learning performance over time

## 🐛 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Ensure API keys are properly set in environment variables
   - Check API key validity and permissions

2. **Document Upload Issues**
   - Verify file formats (PDF, TXT only)
   - Check file size limits (200MB max)
   - Ensure files are not corrupted

3. **Course Generation Failures**
   - Try reducing the number of modules
   - Check internet connectivity
   - Verify API quotas and limits

4. **Memory Issues**
   - Clear browser cache
   - Restart the application
   - Reduce concurrent operations

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
