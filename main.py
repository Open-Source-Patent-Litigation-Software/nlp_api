from typing import Union
import uvicorn
from nlp import NaturalLanguageProcessing
from fastapi import FastAPI
from pydantic import BaseModel

# uvicorn main:app --reload

app = FastAPI()

nlp = NaturalLanguageProcessing()

print("started")

class ScoreInput(BaseModel):
    metrics: list[str]
    text: str

class ScoreOutput(BaseModel):
    scores: list[float]

# Define the Pydantic model for the input data
class CitationsInput(BaseModel):
    description: str
    abstract: str
    claims: str
    metrics: list[str]

class Citation(BaseModel):
    metric: str
    description: list[str]
    abstract: list[str]
    claims: list[str]

# Define the Pydantic model for the output data (optional, for clarity)
class CitationsOutput(BaseModel):
    data: list[Citation]


@app.get("/")
def read_root():
    return {"Response": "World"}

@app.post("/citation", response_model=CitationsOutput)
async def citation(input_data: CitationsInput):
    # generate the citations for these metrics
    citations = nlp.GenerateCitations(input_data.metrics, input_data.abstract, input_data.description, input_data.claims)

    response = {
        "data": citations,
    }
    return response

@app.post("/score", response_model=ScoreOutput)
async def score(input_data: ScoreInput):
    # generate the citations for these metrics
    scores = nlp.generateSimilarityScore(input_data.metrics, input_data.text,)

    response = {
        "scores": scores,
    }
    return response

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)