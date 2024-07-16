import pytest
from pydantic import ValidationError
from models import ScoreInput, ScoreOutput, CitationsInput, CitationsOutput, Citation

def test_score_input():
    # Valid input
    valid_input = {
        "metrics": ["accuracy", "precision", "recall"],
        "text": "Sample input text"
    }
    score_input = ScoreInput(**valid_input)
    assert score_input.metrics == valid_input["metrics"]
    assert score_input.text == valid_input["text"]

    # Invalid input: missing 'metrics'
    invalid_input = {
        "text": "Sample input text"
    }
    with pytest.raises(ValidationError):
        ScoreInput(**invalid_input)

def test_score_output():
    # Valid input
    valid_input = {
        "scores": [0.9, 0.8, 0.75]
    }
    score_output = ScoreOutput(**valid_input)
    assert score_output.scores == valid_input["scores"]

    # Invalid input: scores are not floats
    invalid_input = {
        "scores": ["high", "medium", "low"]
    }
    with pytest.raises(ValidationError):
        ScoreOutput(**invalid_input)

def test_citations_input():
    # Valid input
    valid_input = {
        "description": "Sample description",
        "abstract": "Sample abstract",
        "claims": "Sample claims",
        "metrics": ["novelty", "inventiveness"]
    }
    citations_input = CitationsInput(**valid_input)
    assert citations_input.description == valid_input["description"]
    assert citations_input.abstract == valid_input["abstract"]
    assert citations_input.claims == valid_input["claims"]
    assert citations_input.metrics == valid_input["metrics"]

    # Invalid input: missing 'description'
    invalid_input = {
        "abstract": "Sample abstract",
        "claims": "Sample claims",
        "metrics": ["novelty", "inventiveness"]
    }
    with pytest.raises(ValidationError):
        CitationsInput(**invalid_input)

def test_citations_output():
    # Valid input
    valid_input = {
        "data": [
            {
                "metric": "novelty",
                "description": ["desc1", "desc2"],
                "abstract": ["abs1", "abs2"],
                "claims": ["claim1", "claim2"]
            }
        ]
    }
    citations_output = CitationsOutput(**valid_input)
    assert len(citations_output.data) == 1
    citation = citations_output.data[0]
    assert citation.metric == valid_input["data"][0]["metric"]
    assert citation.description == valid_input["data"][0]["description"]
    assert citation.abstract == valid_input["data"][0]["abstract"]
    assert citation.claims == valid_input["data"][0]["claims"]

    # Invalid input: 'metric' is not a string
    invalid_input = {
        "data": [
            {
                "metric": 123,
                "description": ["desc1", "desc2"],
                "abstract": ["abs1", "abs2"],
                "claims": ["claim1", "claim2"]
            }
        ]
    }
    with pytest.raises(ValidationError):
        CitationsOutput(**invalid_input)

if __name__ == "__main__":
    pytest.main()
