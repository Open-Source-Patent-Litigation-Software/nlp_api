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

class Quote(BaseModel):
    before: str
    highlight: str
    after: str

class Citation(BaseModel):
    metric: str
    description: list[Quote]
    abstract: list[Quote]
    claims: list[Quote]

class CitationsOutput(BaseModel):
    data: list[Citation]