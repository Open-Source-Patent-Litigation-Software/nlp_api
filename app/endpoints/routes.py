from fastapi import APIRouter, Depends
from utils.validators import CitationsInput, ScoreInput, ScoreOutput, RankScoresOuput, RankScoresInput
from nlp.nlp import NaturalLanguageProcessing, getNLP
from utils.auth import get_api_key

router = APIRouter()

@router.post("/citation", dependencies=[Depends(get_api_key)])
def citation(input_data: CitationsInput, nlp: NaturalLanguageProcessing = Depends(getNLP)):
    # generate the citations for these metrics
    citations = nlp.GenerateCitations(input_data.metrics, input_data.abstract, input_data.description, input_data.claims)

    response = {
        "data": citations,
    }
    return response

@router.post("/score", response_model=ScoreOutput, dependencies=[Depends(get_api_key)])
def score(input_data: ScoreInput, nlp: NaturalLanguageProcessing = Depends(getNLP)):
    # generate similarity scores for metrics
    scores = nlp.generateSimilarityScore(input_data.metrics, input_data.text)

    response = {
        "scores": scores,
    }
    return response

@router.post("/rankScores", response_model=RankScoresOuput, dependencies=[Depends(get_api_key)])
def rankScores(input_data: RankScoresInput, nlp: NaturalLanguageProcessing = Depends(getNLP)):
    # generate similarity scores for metrics
    texts = []

    for index in range(len(input_data.abstracts)):
        texts.append(input_data.abstracts[index] + '\n' + input_data.claims[index])
    scores = []
    for text in texts:
        score = nlp.generateSimilarityScore(input_data.metrics, text)
        scores.append(score)

    response = {
        "scores": scores,
    }

    print("finished request")
    return response

@router.post("/similarCPC", response_model=RankScoresOuput, dependencies=[Depends(get_api_key)])
def SimilarCPC(input_data: RankScoresInput, nlp: NaturalLanguageProcessing = Depends(getNLP)):
    print("hello world")
