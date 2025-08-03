import pytest

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