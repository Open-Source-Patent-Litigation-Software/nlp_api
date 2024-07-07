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

    def remove_stopwords(self, text: str) -> str:
        """Remove stopwords from the text"""
        word_tokens = word_tokenize(text)
        filtered_sentence = [w for w in word_tokens if w.lower() not in self.stop_words]
        return " ".join(filtered_sentence)

    @staticmethod
    def split_into_sentences(text: str) -> list:
        """Split text into sentences using regular expressions."""
        text = text.replace("\n", " ")
        sentences = re.split(r"(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s", text)
        sentences = [sentence.strip() for sentence in sentences if sentence]
        return sentences

    def embed_texts(self, texts: list):
        """Embed a list of texts into numerical vectors using the sentence transformer model"""
        embeddings = self.model.encode(texts)
        return embeddings

    def compare_texts_parallel(self, text1: str, sentences: list):
        """Compare the similarity between text1 and a list of sentences"""
        vector1 = self.embed_texts([text1])[0]
        sentence_embeddings = self.embed_texts(sentences)

        similarities = cosine_similarity([vector1], sentence_embeddings)[0]
        max_similarity = np.max(similarities)

        return max_similarity

    def run_similarity_max_calc(self, metric: str, data: str, type: bool):
        """Run the similarity calculation based on the metric provided"""

        cleaned_data = self.remove_stopwords(data)
        sentences = self.split_into_sentences(cleaned_data)
        max_similarity = self.compare_texts_parallel(metric, sentences)
        return max_similarity


def main():
    # Create an instance of the NaturalLanguageProcessing class
    nlp = NaturalLanguageProcessing()

    metric = "A coffee machine."
    data = """CROSS REFERENCE TO RELATED APPLICATIONS\nThe present application is a National Stage of International Application No. PCT/EP2017/064025, filed on Jun. 8, 2017, which claims priority to European Patent Application No. 16173954.5, filed on Jun. 10, 2016, the entire contents of which are being incorporated herein by reference.\nFIELD OF INVENTION\nThe present invention relates to a nitrogen infused soluble cold brew instant coffee and processes for its preparation. In particular, the invention relates to dried coffee powders with improved properties of flavour and stability.\nBACKGROUND\nToday nitrogen gas is widely used to store and dispense carbonated beverages such as beer and soda. Coffee infused with nitrogen (for example “nitro coffee”) has become a popular beverage recently. Nitro coffee is a typically a cold-brew coffee with dissolved nitrogen. Cold brew coffee is considered a low yield coffee with yields in the range of 10-15% and with coffee solids in the range of 0.5-1.5%. This cold beverage is very low in calories, contains no added sugar or alcohol and is entirely a natural product. A key aspect of such chilled beverage is the foam. Nitrogen bubbles show a spectacular cascading effect in cold coffee by first appearing through nucleation of the dissolved gas and sinking to bottom of the receiver (a mug or glass) followed by ascending to the surface while eventually forming an indulgent creamy foam layer. Coffee aromas in concentrates or ready-to-drink (RTD) is known to be not very stable, hence delivering cold brew high quality aroma is challenging. Furthermore for nitro coffee available today, a cold brew preparation must be done shortly before nitro coffee preparation, this involves an added efforts on breweries, coffee shops and/or bars to prepare a stock of liquid cold brew first which is typically an overnight process. It would be advantageous to have a cold brew in instant powder form which can be prepared anytime instantly. Instant coffee is soluble coffee powder that can be dissolved in water to provide a fast and convenient way for consumers to prepare coffee.\nCoffee is typically prepared by brewing roasted and ground coffee beans in hot water. The flavour characteristics of the coffee are influenced by many factors, including the roasting conditions, the size of the ground particles, and the time in which the coffee grounds are in contact with the hot water during brewing.\nInstant coffee can be produced by drying such a brew to form a powder; a typical drying method is freeze drying.\nWhile instant coffee is valued by consumers for its convenience, it is known that such soluble coffee powders often have flavour characteristics that are different to those of freshly brewed coffee. Instant coffee is usually perceived as being less fresh by consumers who like fresh brewed coffee.\nAs consumers show increasing preference for freshly-brewed or even cold-brewed coffees, there is considerable commercial interest in the development of instant coffees that have improved flavour characteristics that more closely replicate the experience of drinking premium freshly brewed coffees, but which can advantageously be marketed as stable dried coffee powder in the existing format of a coffee jar. Such improved instant coffees would enable the targeting of consumers who prefer the taste of ultra-high quality fresh brewed or cold brewed coffee but like the convenience of instant coffee.\nDue to the chemistry of freshly brewed, premium coffee, it is very difficult to dry it and make a sufficiently stable powder. A freeze dried powder prepared from a premium coffee brew can be highly hygroscopic (it attracts and absorbs water) with a tendency to form a “cake” or collapse in the jar.\nA number of approaches have been used to try to address this problem.\nApart from nitro coffee described above, nitrogen is also used in trace amounts in packaged beverages to replace oxygen while packaging. WO2014176102 describes such an aseptic hot-brewed package coffee or expresso beverage containing trace amounts of nitrogen to replace oxygen in the package for increasing shelf-life of the beverage. EP0745329 described a carbonated coffee beverage which has been packaged under pressure in a pressure-resistant closed container, which beverage is based on coffee extract, and wherein the coffee beverage has been packaged in the closed container in the presence of CO2 and nitrogen.\nDuring conventional soluble coffee manufacturing, the extraction is done in two steps, the first extract is produced at or about boiling water temperatures and has flavour characteristics that are brew-like. The second extract from the pre-extracted grounds is produced at higher temperatures approximately 160° C.-204° C. and has strong bitter and “processed” flavour characters.\nHowever, while the first extract has desirable characteristics, it is difficult to make a stable freeze dried coffee powder using this extract only because of its deficiency in high molecular weight compounds. Therefore to make an instant coffee powder that has good stability, both extracts are combined, thereby sacrificing some positive sensory attributes.\nAn alternative approach to making a stable instant coffee powder using only the first extract is to add bulking agents such as maltodextrins. However, with this method the coffee powder produced is no longer “100% coffee” and cannot be labelled as such.\nThere is therefore a need in the art for improved soluble instant coffee powders and processes for their production, that do not suffer from the above-described drawbacks.\nSUMMARY OF THE INVENTION\nThe present invention addresses the above prior art problems (freshness and stability) by providing soluble instant coffee and processes for its preparation, as specified in the claims.\nIn one aspect, the invention provides a liquid coffee beverage comprising liquid coffee having anhydrocarbohydrate content between 10 and 20% w/w and nitrogen gas, wherein the liquid coffee is obtained from a soluble coffee powder. The advantage of using soluble coffee powder is that a liquid coffee can be prepared immediately and nitrogen gas can be infused to this mixture. This process avoids the cumbersome process of producing a cold brew coffee with long brewing periods lasting from 8 to 24 hours to prepare a liquid coffee.\nIn one embodiment the soluble coffee powder is obtained by a process comprising the steps of: (i) extracting coffee solids from roasted and ground coffee beans using water at a temperature of between 0 and 110° C. to obtain a first coffee extract; (ii) filtering the first coffee extract using a selectively-permeable membrane to reduce the concentration of low molecular weight components and provide a filtered coffee extract; (iii) drying the filtered coffee extract to form a dried coffee powder.\nIn another embodiment the soluble coffee powder is obtainable from a dried roast and ground coffee product comprising roast and ground coffee particles that are infused and/or coated with at least 10 weight % of soluble coffee solids and wherein said soluble coffee solids have been extracted at a temperature below 60° C. In one embodiment the dried roast and ground coffee product is blended with roast and ground coffee beans that were not infused and/or coated with soluble coffee solids or blended with spent ground coffee and/or with micronized roasted coffee.\nIn one embodiment of the present invention, the nitrogen is a pure nitrogen gas having at least 99.5% N2.\nIn one aspect of the present invention, the dried roast and ground coffee product defined above comprises the steps of:\n\n\n\na) extracting roast and ground coffee beans at a temperature below 60° C.;\nb) cooling the coffee extract of step a) to a temperature between 4° C. and 10° C.;\nc) mixing the cooled coffee extract of step b) with roast and ground coffee that was not extracted in step a) in a ratio of soluble coffee solids to roast and ground coffee between 1:1 and 1:10, thereby infusing and/or coating the roast and ground coffee particles with soluble coffee solids; and\nd) drying the infused and/or coated roast and ground coffee of step c).\n\n\n\n\nIn another aspect of the present invention, the dried roast and ground coffee product defined above comprises the steps of\n\n\n\na) mixing roast and ground coffee with water at temperature below 60° C., in a ratio of roast and ground coffee to water between 1:1 and 1:5, thereby obtaining a slurry;\nb) extracting roast and ground coffee of the slurry in a vacuum chamber, by applying a pressure between 75 mbar and 400 mbar for between 1 and 12 minutes at temperature between 10° C. and 35° C.; and\nc) drying the slurry of step b).\n\n\n\n\nMoreover, it was suprisingly found that nitrogen gas infused in to such a filtered coffee extract preparation provides a better foam volume which is stable over time.\nIn one embodiment, the water is at a temperature of between 0 and 100° C. (e.g. between 20 and 50° C., between 10 and 40° C., between 20 and 40° C., or between 20 and 30° C.).\nIn one embodiment, the water is at a temperature of about 0, 10, about 15, about 20, about 25, about 30, about 35, about 40, about 45, 50, 60, 70, 80, 90, 100, 110° C.\nIn one embodiment, the coffee extract is passed through a membrane that has a molecular weight cut off of 0.1-100 kDa. The membrane can be of organic or inorganic material.\nIn one embodiment, the dried coffee powder comprises a ratio of coffee compounds where the concentration of high molecular weight compounds to the concentration of low molecular weight compounds (as defined by the size exclusion chromatography technique) is at least 5.\nIn one embodiment, the filtered coffee extract is concentrated prior to drying, for example by reverse osmosis or low temperature vacuum evaporation.\nIn one embodiment, the filtered coffee extract is dried using freeze drying, vacuum belt drying or spray drying.\nIn one embodiment, coffee aroma is recovered prior to filtration and is subsequently blended with the filtered coffee extract prior to drying.\n\n\n\nDESCRIPTION OF FIGURES\nFIG. 1 presents a flow-chart showing an example process according to the invention.\nFIG. 2 presents an SEC chromatogram of a coffee extract produced using water at a temperature of 25° C.\nFIG. 3 presents a measurement of foam volume vs time using nitrogen infused coffee extract of the present invention.\n\n\n\nDETAILED DESCRIPTION OF THE INVENTION\nAccording to the present invention the term “beverage” means any noncarbonated aqueous liquid material that is a homogeneous liquid substantially free of solids having a flavor due to dissolved components.\nAccording to the present invention dispensing of the chilled beverage means opening a faucet/draft column of the system to allow the chilled “nitrogen infused” beverage to flow from the system into a receiver such as a glass, mug or other drinking container. Throughout the following description the term “nitrogen infused” will be used to describe a nitrogen rich coffee beverage having either N2 or N2O or N2/C02 or N2/N2O/C02 infused beverage. If an embodiment is directed specifically to a N2/C02 mixture or specifically to only N2 infusion, the actual gas composition is explicitly disclosed.\nDispensing of the nitrogen infused chilled beverage is an element of the present invention wherein reduction of pressure on the gas infused beverage allows the nucleation of the dissolved gas producing micro bubbles resulting in unique properties which distinguish the dispensed beverage by enhancement of the beverage's flavor and/or appearance. For instance appearance of foam and stability of foam over time and taste and aroma of coffee delivered through this beverage.\nThe term “anhydrocarbohydrate” refers to carbohydrate distribution of essentially mannose, arabinose and galactose. The total content ranges from 10 to 20% w/w. In one embodiment, the carbohydrate distribution of the coffee of the present invention may comprise for instance about 15.7 w/w % which comprises essentially 6.1% mannose; 6% galactose and 2.6% arabinose. In another embodiment, the carbohydrate distribution of the coffee of the present invention may comprise for instance about 10.7 w/w % which comprises essentially 3.3% mannose; 4% galactose and 3% arabinose. The anhydrocarbohydrate content is determined through high pressure chromatography using anion exchange stationary phase with amperometric detection and after compete hydrolysis fo the sample. Carbohydrate molecular weight distribution was performed using size exclusion chromatography. Then in line hydrolysis with sulfuric acid adding 3,5 dihydroxytoluen with colorimetric detection. Response is therefore proportional to total carbohydrates monomers as 3,5 dihydroxytoluen is selective with carbs.\nThe present invention provides a dried coffee powder obtainable by a process comprising membrane filtration of a low temperature extract of roasted and ground coffee beans to reduce the concentration of low molecular weight components and drying the filtered coffee extract.\nThe present inventors have discovered that a dried coffee powder suitable for use as an instant coffee and having highly desirable flavour characteristics of brewed coffee can be prepared by using membrane filtration to reduce the concentration of low molecular weight (LMW) components in a brewed coffee extract prior to a drying process. The dried coffee powder produced has good stability characteristics and low hygroscopicity, thus enabling it to be stored for long periods of time and making it suitable for use as an instant coffee. Morover, it was suprisingly found that nitrogen gas infused into and extract produced form the dried coffee powder provides a better foam volume which is stable over time.\nThe process of membrane filtration reduces the concentration of LMW components and provides a concomitant increase in the ratio of high molecular weight (HMW) components to LMW components in the filtered coffee extract.\nBy using membrane filtration to reduce the concentration of low molecular weight components in the brewed coffee extract, a stable instant coffee powder.\nExtraction is the process by which coffee solids (e.g. soluble coffee solids) are extracted from roasted and ground coffee beans, typically using water, to form a solution referred to as a coffee extract.\nThe process of the invention is carried out using a low temperature extract of roasted and ground coffee beans. As used herein, the term “low temperature extract” is preferably a coffee extract obtained using water at a temperature of between 0 and 110° C.\nThe process of the invention uses membrane filtration to reduce the concentration of low molecular weight components of the coffee extract. Thus, the coffee extract is passed over a membrane which is selectively permeable to LMW components of the coffee extract, allowing these to be separated and thus reducing their concentration in the coffee extract. By reducing the concentration of LMW components, there is a concomitant increase in the ratio of HMW to LMW components.\nIn a preferred embodiment, the term “low molecular weight component” refers to compounds present in a coffee extract (coffee solids) that have a molecular weight of less than about 1 kDa (for example, less than about 0.9, 0.8, 0.7, 0.6 or 0.5 kDa), and the term “high molecular weight component” refers to compounds present in a coffee extract (coffee solids) that have a molecular weight greater than about 1kDa (for example, greater than about 1.1, 1.2, 1.3, 1.4 or 1.5 kDa).\nThe present inventors have discovered that a dried coffee powder having particularly advantageous properties (such as advantageous stability properties) is produced when the ratio of HMW to LMW components (as defined by the size exclusion chromatography technique) is at least 5 (for example, at least 5, at least 5.5, at least 6, at least 6.5, or at least 7).\nThe present invention provides a process for preparing a dried coffee powder, said process comprising membrane filtration of a low temperature extract of roasted and ground coffee beans to reduce the concentration of low molecular weight components and drying the filtered coffee extract.\nIn a preferred embodiment, the process comprises the steps of: (i) extracting coffee solids from roasted and ground coffee beans using water, preferably at a temperature of between 0 and 110° C., to obtain a first coffee extract; (ii) filtering the first coffee extract using a selectively-permeable membrane to reduce the concentration of low molecular weight components, wherein low molecular weight coffee solids that pass through the filter form a permeate. The high molecular weight coffee solids that are retained by the filter form a retentate; and (iii) drying the retentate to form a dried coffee powder.\nThe process of the present invention comprises a low temperature extraction using water, preferably at a temperature of between 0 and 110° C., to obtain a first coffee extract.\nIn one embodiment, the water is at a temperature of between 0 and 110° C. (for example, between 20 and 50° C., between 10 and 40° C., between 20 and 40° C., or between 20 and 30° C.). In one embodiment, the water is at a temperature of about 10, about 15, about 20, about 25, about 30, about 35, about 40, about 45, or about 50° C.\nThe roasted coffee beans are ground prior to extraction. Any suitable beans may be used. Methods for roasting and grinding coffee beans to obtain desired characteristics are well known in the art.\nThe extraction may be carried out in any suitable extraction vessel, for example fixed bed reactors or continuous counter-current extractors.\nThe extraction yield of a coffee extract refers to the percentage of coffee solids that are transferred (i.e. extracted) to the water during the extraction step. The extraction yield can be controlled using extraction water temperature and the ratio of water to beans. The inventors have found that coffee extracts produced with a low yield provide particular advantageous flavour characteristics when used in the process of the invention.\nFollowing the extraction, a “first coffee extract” is obtained. The first coffee extract is filtered using a membrane, which enables a reduction in the concentration of LMW components (for example, components have a molecular weight less than about 1 kDa). The membrane is selectively permeable with a molecular weight cut-off value that allows only LMW components to pass through the membrane. In one embodiment, the membrane has a molecular weight cut-off of 1 kDa, meaning that compounds having a molecular weight of greater than about 1kDa are retained by the membrane.\nThus, coffee solids having a molecular weight less than the molecular weight cut-off value of the membrane (i.e. LMW components of the coffee extract) are able to pass through the filter while coffee solids having a molecular weight greater than the molecular weight cut-off value of the membrane (i.e. HMW components of the coffee extract) are unable to pass through the filter and are therefore retained in the coffee extract. Filtration using such a selectively-permeable membrane therefore separates the coffee extract into two different fractions: the LMW fraction that passes through the filter is referred to as the permeate, while the HMW fraction that is retained by the filter is referred to as the retentate.\nThe permeate may be alternatively recycled for use in a separate coffee product.\nIn one embodiment, the filtered coffee extract comprises a ratio of HMW components to LMW components of at least 5.\nAs discussed above, the present inventors have discovered that a dried coffee powder having particularly advantageous properties (such as stability properties) is produced when the concentration ratio of HMW components (for example, those having a molecular weight greater than about 1kDa) to the concentration of LMW components (for example, those having a molecular weight less than about 1kDa) is at least 5 (e.g. at least 5, at least 5.5, at least 6, at least 6.5, or at least 7).\nTo improve the efficiency of the filtration process, the retentate may be recycled and subjected to the filtration process multiple times.\nThe filtration step may be carried out using cross-flow filtration, in which the fluid flow is tangential to the surface of the membrane, or using “dead end” filtration, where the fluid flow is perpendicular to the membrane or any other membrane fractination technique.\nMembranes suitable for use in the process of the invention include nanofiltration membranes having a molecular weight cut-off of 0.1-100 kDa.\nSpecifications of an example suitable membrane are as follows:\n\n\n\n\n\n\n\nTABLE 1\n\n\n\n\n\n\n\nAn example of membrane properties\n\n\n\n\n\n\n\n\n\n\nParameter\nSpecification\n\n\n\n\n\n\n\nsucrose rejection at 70-145 psi\n45-75%\n\n\n\nNaCl rejection at 70-145 psi\n50-60%\n\n\n\nOperating pH range\n2-11 \n\n\n\nCleaning pH range\n1-12 \n\n\n\n\n\n\n\n\n\n\n\nMaximum cleaning temperature\n50°\nC.\n\n\n\nTypical operating temperature\n5-50°\nC.\n\n\n\nMaximum chlorine concentration\n<100\nppm\n\n\n\n\n\n\n\n\n\n\nSuitable membrane sizes will vary depending on the scale of the production process.\nA diafiltration step may be carried out in combination with the filtration step. A diafiltration step consists of adding dilution water to the retentate product then removing a permeate fraction equivalent to the amount of added dilution water.\nFollowing the filtration step, the retentate (i.e. the filtered coffee extract) is dried to form a soluble coffee powder.\nSuitable processes for drying a coffee extract to produce a soluble coffee powder (an instant coffee) are known in the art and include freeze drying and spray drying. Thus, in one embodiment, the filtered coffee extract is freeze dried to form a dried coffee powder. In one embodiment, the filtered coffee extract is spray dried to form a dried coffee powder.\nIn a freeze drying process, a liquid coffee extract is frozen at about −20° C. to about −40° C., before being heated under low pressure conditions. Application of low pressures enables the frozen water component to be removed (such as by sublimation) without needing high temperatures, which could degrade the flavour and other characteristics of the coffee extract.\nSpray drying is an alternative to freeze drying. In spray drying, a liquid coffee extract is sprayed through a small nozzle into a heated drying gas. This produces dried coffee particles which can subsequently be collected.\nThe process of the invention may comprise an additional concentration step prior to the drying step. Such a concentration step can be used to increase the strength of the coffee extract and improve the flavour characteristics. Thus, in one embodiment, the filtered coffee extract (the retentate) is concentrated prior to drying, optionally by reverse osmosis or low temperature vacuum evaporation, freeze concentration, or any other technique known in the art.\nThe aroma of coffee comes from multiple different chemical compounds which make up the aroma components. Coffee aroma is an important quality which can influence the taste and perception of coffee by consumers. If a coffee product lacks the aroma commonly associated with it, consumer perception of the coffee may be adversely affected. This can be a problem in the field of instant coffee, where processes of extraction, concentration and drying may reduce or remove the coffee aroma. For these reasons, it may be advantageous to recover coffee aromas which are given off during the processing of the coffee and to reintroduce these aromas to the coffee extract prior to drying.\nThus, in one embodiment, coffee aroma is extracted from the roasted and ground coffee beans prior to the extraction of the coffee solids, and said coffee aroma is subsequently blended with the filtered coffee extract prior to drying.\nProcesses for extracting coffee aromas and subsequently reintroducing them to coffee extracts prior to drying are known in the art. An example of a suitable process is vacuum extraction (VAX). Processes for recovering coffee aromas are described in WO 1999/052378 and WO 2001/013735.\nThe dried coffee powder of the invention has good stability properties, making it suitable for use as an instant coffee. Instant coffee is typically packaged and sold in jars, which are stored at room temperature for long periods of time. Thus, in one embodiment, the dried coffee powder of the invention is advantageously stable at room temperature for at least six months.\nThose skilled in the art will understand that they can freely combine all features of the present invention described herein without departing from the scope of the invention as disclosed.\nEXAMPLES\nVarious preferred features and embodiments of the present invention will now be described by way of non-limiting examples.\nExample 1\nPreparation of a Dried Coffee Powder.\nAn example flow-chart is shown in FIG. 1.\nCoffee beans were roasted and ground. Coffee aroma was extracted from the roasted and ground beans using a vacuum extraction process; the coffee aroma was stored for later reintroduction into the process.\nThe aroma-extracted roasted grounds were introduced into an extractor and the coffee solids extracted using water at a temperature of 25° C. using water to coffee ratio of 4.0.\nThe above obtained coffee extract then underwent a nanofiltration process using a semi-permeable membrane with a molecular weight cut-off of 1kDa. Low molecular weight components were filtered into permeate, while high molecular weight components remained in the retentate.\nThe filtered coffee extract in the form of the retentate comprised the concentration of HMW and the concentration of LMW components at a ratio of approximately 5 (HMW to LMW, as defined by the SEC technique).\nThe filtered coffee extract was freeze dried to produce a stable dried coffee powder.\nExample 2\nPreparation Using Different Membranes\nVarious polymeric ultrafiltration (UF) and nano filtration (NF) membranes with molecular weight cut offs (MWCO) ranging between 500 and 20,000 Da were used to fractionate cold brew coffee extract during a series of preliminary bench scale screening experiments. The fractionations were conducted on a plate-and-frame unit equipped with 84 cm2 of membrane area and operated at temperatures between 10-60° C. and pressures up to 30 bars.\nThe results from the initial screening tests identified NF membrane (molecular weight cut-off of 1 kDa) as a particularly suitable membrane for the fractionation of low temperature coffee extract. The selection of the membrane was based on its performance in terms of permeation flux, cleanability and its ability to fractionate and separate enough of the low molecular weight coffee compounds in the permeate so that the obtained retentate can be used to make a dryable product with superior sensory qualities.\nExample 3\nPreparation of Prototype Dried Coffee Powders.\nExtraction trials were conducted using pilot plant scale extraction. To evaluate the impact of temperature on the extracts, three trials were done to produce coffee extracts at temperatures of 25°, 50° and 85° C.\nThe resulting extract was fractionated using a 2-module co-pilot scale nanofiltration system (11 m2 membrane area) operated in the batch mode at ambient temperature. A concentration factor (CF) of 4.0 was achieved during initial fractionation. The retentate was further subjected to diafiltration to wash out more of the low molecular weight compounds from the retentate. The diafiltration step consisted of adding dilution water to the retentate product then removing a permeate fraction equivalent to the amount of added dilution water.\nThe results of the membrane performance in terms of permeation flux, solids recovery are summarized in Table 2.\n\n\n\n\n\n\n\nTABLE 2\n\n\n\n\n\n\n\nMembrane performance for producing coffee extract fractions\n\n\nPermeation flux performance during nanofiltration of pre-E1 extracts\n\n\n\n\n\n\n\n\n\n\n\nSingle cell pilot plant extraction\n\n\n\nProcessing\nTrial #\n\n\n\n\n\n\n\n\n\n\n\n\n\nconditions\nTrial 1\nTrial 2\nTrial 3\nTrial 4\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\n\nExtraction\nTemp. ° C.\n85\n50\n25\n25\n\n\n\nTotal Solids\n4.51\n3.9\n2.41\n2.1\n\n\nNanofiltration\nApplied pressure\n17-30\n8 to 26\n10 to 20\n10 to 18\n\n\n\n(bar)\n\n\n\nAvg Flux,\n6.1\n7.8\n8.3\n8.2\n\n\n\nL/m2 · h\n\n\n\nAvg. diaf. Flux\n6.8\n6.8\n7.3\n7.5\n\n\n\n\n\n\n\n\n\nAroma was added to the diafiltered retentate to make the aromatized freeze-dried powders.\nExample 4\nSize-Exclusion Chromatography (SEC) Analysis.\nSize-exclusion chromatography was used to analyse filtered coffee extracts produced using the process of the invention.\nCoffee extract was separated using two size exclusion columns (Superose 60 and Superdex Peptide from GE Healthcare) connected in series in an HPLC (high performance liquid chromatography). Water was used as the mobile phase at a flow rate of 0.5 ml/minute. The peaks were visualized using a refractive index detector. The run time for the chromatogram was 120 minutes.\nIt was possible to visualize the compounds forming the coffee extract. The SEC chromatogram showed two distinct peak clusters (FIG. 3). The material eluting before 60 minutes comprised high molecular weight compounds, while the peak cluster eluting after 60 minutes was low molecular weight compounds.\nThe peak area under the two peak clusters was determined. The ratio of the peak areas for the high molecular weight (HMW) materials peaks and the low molecular weight (LMW) material peaks was calculated.\nA coffee extract produced for use in the process of the invention (25° C.) was analyzed using this technique. The SEC chromatogram was as shown in FIG. 3. The high molecular weight peak cluster and the low molecular weight peak clusters were visualized and the ratio of the high molecular weight compounds to low molecular weight compounds was calculated to be 5.0\nThe retentate produced using membrane with diafiltration, was further freeze dried into a dried coffee powder. The filtered extract formed a relatively stable powder when compared with the powder produced from the extract without to membrane filtration.\nExample 5\nReference Sample\nA coffee beverage was prepared by reconstituting a soluble coffee powder infused with nitrogen as described in WO2009/040249. This powder was used as a reference.\nA liquid composition was obtained using 1.3 wt % of the stable powder made of N2 loaded granules of soluble coffee, dipsersed into 4° C. water. Dissolution is bad and some lumps visible both in the beverge and in the foam. Foam level is very low and bubbles are very polydisperse.\nExample 6\nPreparation of Nitogen Gas Infused Cold Coffee Beverage.\nA coffee solution of 30 liters, dosing at 1.3% coffee solids, in cold water was prepared using the stable powder as described in example 1. This solution was placed in a keg under pressure at 3-4 bar using Nitrogen gas. The Keg was placed in a cold room at 4-8C for 48 hours. The pressure was checked regularly to ensure minimum of 3 bars. After 48 hours, the keg was connected to a standard beer tap and to the nitrogen bottle to release the liquid through the beer tap. The beverage was served in a glass mug. A nice foamy and creamy beverage with cascading of foam was recorded.\nBeverage Characterization\nBeverage is dispensed through faucet in the form of an homogeneous foam made of fine bubbles dispersed homogeneously all over the beverage.\nAfter beverage production, the bubbles are instantaneously creaming due to density difference between air and continuous liquid phase.\nAfter 3 minutes, a large majority of bubble has creamed forming a foam layer on the top of the beverage: the coffee crema.\nCoffee crema evolves over time due to bubble coalescence, Ostwald ripening and liquid drainage.\nIn order to characterize the beverage, photometry was used. A photographic picture of the sample is made using CoffeeCam (Newtone Technonogies, France) from top and/or side view in a controlled light environment followed by a robust and accurate image analysis in the CIE Lab colorimetric space.\nIn the specific case of layer detection (ie a coffee crema on top of a liquid coffee phase), a layer can be considered as a discontinuity in colour in the beverage.\nFoam is also characterized regarding its texture with a standard Rheometer (Discovery HR2, TA Instruments, US) with a cup and vane geometry. A flow curve from 0.1 up to 100 s-1 is performed.\nHigh shear viscosity and yield stress are used to defined foam rheology.\nFIG. 3 shows foamability and foam stability measurement for the beverage of the present invention.", "pn": "US11266160B2"""

    print(nlp.run_similarity_max_calc(metric, data))


# Entry point: Check if the script is being run directly (not imported as a module)
if __name__ == "__main__":
    # Call the main function to execute the code
    main()
