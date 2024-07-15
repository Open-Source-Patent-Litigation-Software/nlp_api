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

    def __embed_texts(self, texts: list):
        """Embed a list of texts into numerical vectors using the sentence transformer model"""
        embeddings = self.model.encode(texts)
        return embeddings

    def __compare_texts_parallel(self, text1: str, sentences: list):
        """Compare the similarity between text1 and a list of sentences"""
        vector1 = self.__embed_texts([text1])[0]
        sentence_embeddings = self.embed_texts(sentences)

        similarities = cosine_similarity([vector1], sentence_embeddings)[0]
        print(similarities)
        max_similarity = np.max(similarities)

        return max_similarity

    def __run_similarity_max_calc(self, metric: str, data: str):
        """Run the similarity calculation based on the metric provided"""

        cleaned_data = self.__remove_stopwords(data)
        sentences = self.__split_into_sentences(cleaned_data)
        max_similarity = self.__compare_texts_parallel(metric, sentences)
        return max_similarity
    
    def __embedSections(self, abstract, claims, description):
        a = self.__embed_texts(self.__split_into_sentences(self.__remove_stopwords(abstract)))
        c = self.__embed_texts(self.__split_into_sentences(self.__remove_stopwords(claims)))
        d = self.__embed_texts(self.__split_into_sentences(self.__remove_stopwords(description)))
        return a, c, d
    
    def __getCitations(self, metric, section_embeddings, sentences):
        """Run the similarity calculation for a single metric and return a list of citations"""
        citations = []

        # Calculate the similarity scores between the metric and section embeddings
        similarities = cosine_similarity([metric], section_embeddings)[0]
        
        # Find sentences that exceed the threshold
        for idx, similarity in enumerate(similarities):
            if similarity > self.threshold:
                citations.append(sentences[idx])
        
        # Remove duplicates
        citations = list(set(citations))

        return citations

    def GenerateCitations(self, metrics: list[str], abstract: str, description: str, claims: str):
        # Generate embeddings for the metrics
        embeddedMetrics = self.__embed_texts(metrics)

        # Split the sections into sentences
        abstractSentences = self.__split_into_sentences(abstract)
        claimsSentences = self.__split_into_sentences(claims)
        descriptionSentences = self.__split_into_sentences(description)

        # Generate embeddings for the sections
        embeddedAbstract, embeddedClaims, embeddedDescription = self.__embedSections(abstract, claims, description) 

        citations_list = []

        # Process citations for each metric
        for metric, embeddedMetric in zip(metrics, embeddedMetrics):
            citations_dict = {
                "metric": metric,
                "abstract": self.__getCitations(embeddedMetric, embeddedAbstract, abstractSentences),
                "claims": self.__getCitations(embeddedMetric, embeddedClaims, claimsSentences),
                "description": self.__getCitations(embeddedMetric, embeddedDescription, descriptionSentences)
            }
            citations_list.append(citations_dict)

        return citations_list

    def generateSimilarityScore(self, metrics, text):
        """ Given a list of metrics and a text (combined abstract + claims), calculate the similarity score for each metric. """

        # Clean and embed the entire text
        cleanedText = self.__remove_stopwords(text)
        sentences = self.__split_into_sentences(cleanedText)
        embeddedText = self.model.encode(sentences)

        similarityScores = []

        # Calculate cosine similarity between each metric and the text embedding
        for metric in metrics:
            cleanedMetric = self.__remove_stopwords(metric)
            embedMetric = self.model.encode([cleanedMetric])
            similarity = cosine_similarity(embedMetric, embeddedText)
            similarityScores.append(np.max(similarity))

        return similarityScores