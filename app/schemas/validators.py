from pydantic import BaseModel

class ScoreInput(BaseModel):
    metrics: list[str]
    text: str

class ScoreOutput(BaseModel):
    scores: list[float]

class RankScoresInput(BaseModel):
    metrics: list[str]
    texts: list[str]

class RankScoresOuput(BaseModel):
    scores: list[list[float]]

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

class CitationsOutput(BaseModel):
    data: list[Citation]