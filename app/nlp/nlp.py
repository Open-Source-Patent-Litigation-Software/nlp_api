from schemas.validators import Citation, CitationsOutput, Quote
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class NaturalLanguageProcessing:
    def __init__(self):
        """Initialize the SentenceTransformer model for Natural Language Processing"""
        nltk.download("stopwords")
        nltk.download("punkt")
        self.model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa")
        self.stop_words = set(stopwords.words("english"))
        self.threshold = 0.6

    def __remove_stopwords(self, text: str) -> str:
        """Remove stopwords from the text"""
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if w.lower() not in self.stop_words]
        return " ".join(filtered_sentence)

    @staticmethod
    def __split_into_sentences(text: str) -> list:
        """Split text into sentences using regular expressions."""
        text = text.replace("\n", " ")
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!|\;)\s", text)
        sentences = [sentence.strip() for sentence in sentences if sentence]
        return sentences
    
    def __split_into_paragraphs(self, text: str) -> list:
        """Split text into paragraphs using regular expressions."""
        paragraphs = re.split(r'\n{2,}', text)
        paragraphs = [paragraph.strip() for paragraph in paragraphs if paragraph]
        return paragraphs

    def __embed_texts(self, texts: list):
        """Embed a list of texts into numerical vectors using the sentence transformer model"""
        embeddings = self.model.encode(texts)
        return embeddings

    def __get_section_embeddings(self, abstract: str, claims: str, description: str):
        """Get embeddings for abstract, claims, and description sections"""
        return (
            self.__embed_texts(self.__split_into_sentences(self.__remove_stopwords(abstract))),
            self.__embed_texts(self.__split_into_sentences(self.__remove_stopwords(claims))),
            self.__embed_texts(self.__split_into_sentences(self.__remove_stopwords(description))),
        )

    def __get_citations(self, metric_embedding, section_embeddings, sentences):
        """Get sentences from sections that exceed the similarity threshold"""
        citations = [
            (similarity, sentence) for similarity, sentence in zip(cosine_similarity([metric_embedding], section_embeddings)[0], sentences)
            if similarity > self.threshold
        ]
        return list(set(citations))  # Remove duplicates

    def __extract_quotes(self, sentences: list[str], highlighted: str):
        index = sentences.index(highlighted)
        before = sentences[index - 1] if index - 1 >= 0 else ""
        after = sentences[index + 1] if index + 1 < len(sentences) else ""
        return Quote(before=before, highlight=highlighted, after=after)

    def GenerateCitations(self, metrics: list[str], abstract: str, description: str, claims: str):
        """Generate citations based on metrics and patent sections"""
        embedded_metrics = self.__embed_texts(metrics)

        # Split the sections into sentences
        abstract_sentences = self.__split_into_sentences(abstract)
        claims_sentences = self.__split_into_sentences(claims)
        description_sentences = self.__split_into_sentences(description)

        # Generate embeddings for the sections
        embedded_abstract, embedded_claims, embedded_description = self.__get_section_embeddings(abstract, claims, description)

        citations = {}

        # Process citations for each metric
        for metric, embedded_metric in zip(metrics, embedded_metrics):
            abstract_citations = [
                self.__extract_quotes(abstract_sentences, citation)
                for _, citation in self.__get_citations(embedded_metric, embedded_abstract, abstract_sentences)
            ]
            claims_citations = [
                self.__extract_quotes(claims_sentences, citation)
                for _, citation in self.__get_citations(embedded_metric, embedded_claims, claims_sentences)
            ]
            description_citations = [
                self.__extract_quotes(description_sentences, citation)
                for _, citation in self.__get_citations(embedded_metric, embedded_description, description_sentences)
            ]

            citation = Citation(
                metric=metric,
                abstract=abstract_citations,
                claims=claims_citations,
                description=description_citations
            )
            citations[metric] = citation

        return citations

    def generateSimilarityScore(self, metrics, text):
        """Given a list of metrics and a text (combined abstract + claims), calculate the similarity score for each metric."""
        paragraphs = self.__split_into_sentences(text)
        cleanedParagraphs = []
        for paragraph in paragraphs:
            cleanedParagraphs.append(self.__remove_stopwords(paragraph))
        embedded_text = self.__embed_texts(cleanedParagraphs)

        similarity_scores = []

        # Calculate cosine similarity between each metric and the text embedding
        for metric in metrics:
            cleaned_metric = self.__remove_stopwords(metric)
            embedded_metric = self.__embed_texts([cleaned_metric])
            similarity = cosine_similarity(embedded_metric, embedded_text)
            similarity_scores.append(np.max(similarity))

        return similarity_scores

nlp = NaturalLanguageProcessing()

def getNLP():
    return nlp