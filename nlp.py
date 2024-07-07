from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math


class NaturalLanguageProcessing:
    def __init__(self):
        # Initialize the sentence transformer model
        self.model = SentenceTransformer("AI-Growth-Lab/PatentSBERTa")

    def embed_text(self, text):
        # Generate embeddings for the text
        embedding = self.model.encode(text)
        return embedding.reshape(1, -1)

    def compare_texts(self, text1, text2):
        # Embed both texts
        vector1 = self.embed_text(text1)
        vector2 = self.embed_text(text2)
        # Calculate and return cosine similarity
        similarity = cosine_similarity(vector1, vector2)[0][0]
        return similarity


def main():
    nlp = NaturalLanguageProcessing()

    text1 = "Automatic shut-off feature."
    text2 = """1. An espresso making device, the device comprising:
a user interface that comprises controls for providing user preferences regarding both a volume size and a strength of a beverage discharged as a coffee dose;
a removable portafilter, the portafilter having a compartment for coffee grounds and a discharge spout; the compartment having an opening for discharging brewed coffee to the spout;
a group head for engaging the portafilter;
an auxiliary port located adjacent to the group head for discharging hot water;
the portafilter defining a secondary flow path that is separate from the compartment and extending from a location near to an auxiliary port when the portafilter is fitted to the group head, to a discharge spout; so as to discharge hot water into the drinking vessel when the vessel is located under the spout;
wherein the secondary flow path has a receiving aperture, in flow communication with the auxiliary port, that is external to the engaged group head and portafilter; such that hot water is selectively discharged into the drinking vessel without passing through the engagement between the group head and the portafilter.
2. The device of claim 1, wherein:
the spout is in fluid flow communication with both the compartment and the secondary flow path to simultaneously deliver a mix of the brewed coffee and the additional hot water.
3. The device of claim 1, wherein:
discharge of the brewed coffee and the additional hot water is controlled by a processor in accordance with the user preferences.
4. The device of claim 3, wherein:
flows are controlled by processor controlled valves in accordance with durations of the discharges determined by the processor.
5. The device of claim 4, wherein:
the brewed coffee discharged is measured by a first flow meter and the additional hot water discharged by the auxiliary port is measured by a second flow meter;
both flow meters providing flow information to the processor from which information the processor independently controls the durations of the discharges.
6. The device of claim 3, wherein:
the brewed coffee discharged and the additional hot water are measured by a pair of flow meters;
the pair of flow meters providing flow information to the processor from which information the processor controls the durations."""

    similarity = nlp.compare_texts(text1, text2)
    print("Similarity score:", math.floor(similarity * 10))


if __name__ == "__main__":
    main()
