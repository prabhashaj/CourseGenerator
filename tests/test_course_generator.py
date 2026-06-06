def test_chapter_details_completeness():
    response = get_chapter_details()  # Replace with actual function to get chapter details
    assert response is not None
    assert 'title' in response
    assert 'content' in response
    assert len(response['content']) > 0
    assert 'tokens_used' in response
    assert response['tokens_used'] <= MAX_TOKENS  # Define MAX_TOKENS as per your application logic

def test_token_limit_handling():
    response = get_chapter_details_with_max_tokens()  # Replace with actual function
    assert response is not None
    assert 'content' in response
    assert len(response['content']) <= MAX_CONTENT_LENGTH  # Define MAX_CONTENT_LENGTH as per your application logic

def test_get_chapter_images_from_tavily(mocker):
    # Mock requests.post
    mock_post = mocker.patch("requests.post")
    
    # Define a mock response containing images
    mock_response = mocker.Mock()
    mock_response.json.return_value = {
        "images": [
            {"url": "https://example.com/image1.jpg", "description": "Diagram 1"},
            "https://example.com/image2.jpg"
        ]
    }
    mock_response.raise_for_status = mocker.Mock()
    mock_post.return_value = mock_response
    
    from course_generator import get_chapter_images_from_tavily
    
    res = get_chapter_images_from_tavily("test query")
    
    assert len(res) == 2
    assert res[0]["url"] == "https://example.com/image1.jpg"
    assert res[0]["description"] == "Diagram 1"
    assert res[1]["url"] == "https://example.com/image2.jpg"
    assert res[1]["description"] == ""