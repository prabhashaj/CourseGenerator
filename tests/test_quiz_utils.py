def test_quiz_utils_token_limits():
    chapter_details = get_chapter_details()  # Assuming this function retrieves chapter details
    assert len(chapter_details) <= MAX_TOKENS
    assert 'title' in chapter_details
    assert 'content' in chapter_details
    assert 'questions' in chapter_details

def test_quiz_utils_complete_details():
    chapter_details = get_chapter_details()
    assert chapter_details['title'] != ''
    assert chapter_details['content'] != ''
    assert len(chapter_details['questions']) > 0