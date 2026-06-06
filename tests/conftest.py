import pytest
import builtins

@pytest.fixture(scope='session')
def token_limit():
    return 4096  # Example token limit

@pytest.fixture(scope='session')
def complete_chapter_details():
    return {
        'title': 'Chapter 1',
        'content': 'This is the complete content of Chapter 1.',
        'tokens_used': 100
    }

# Mock functions and constants for tests
def get_chapter_details():
    return {
        'title': 'Test Chapter',
        'content': 'This is test content',
        'tokens_used': 50,
        'questions': [{'question': 'What?', 'options': ['A', 'B'], 'answer': 'A'}]
    }

def get_chapter_details_with_max_tokens():
    return {
        'content': 'Short content'
    }

builtins.get_chapter_details = get_chapter_details
builtins.get_chapter_details_with_max_tokens = get_chapter_details_with_max_tokens
builtins.MAX_TOKENS = 4096
builtins.MAX_CONTENT_LENGTH = 1000