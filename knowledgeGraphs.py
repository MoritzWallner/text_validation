import spacy
import networkx as nx
import matplotlib.pyplot as plt
from contradictionDetection import check_contradiction

# Load the Spacy model
nlp = spacy.load("en_core_web_sm")

def extract_kg_from_article(text):
    doc = nlp(text)
    kg = nx.DiGraph()

    # Add entities as nodes
    for ent in doc.ents:
        kg.add_node(ent.text, label=ent.label_)
    
    # Use dependency parsing to extract relations
    for token in doc:
        subject = None
        object = None
        negation = ""

        # Find subject and object
        if token.dep_ == 'ROOT':
            # Look for nominal subject (nsubj) and object (dobj, pobj, attr)
            for child in token.children:
                if child.dep_ == 'nsubj':  # Subject
                    subject = child
                if child.dep_ in ('dobj', 'attr', 'pobj'):  # Object
                    object = child
                if child.dep_ == 'neg':  # Handle negation
                    negation = "not "

            # If we have both subject and object, add them to the graph
            if subject and object:
                relation = negation + token.lemma_  # Include negation if present
                kg.add_edge(subject.text, object.text, relation=relation)

    return kg

def generate_sentences_from_kg(graph):
    """Convert the nodes and edges of a KG to simple sentences."""
    sentences = []
    for src, dest, data in graph.edges(data=True):
        relation = data['relation']
        sentence = f"{src} {relation} {dest}."
        sentences.append(sentence)
    return sentences

def visualize_kg(graph):
    plt.figure(figsize=(12, 8))
    
    # Position nodes using spring layout
    pos = nx.spring_layout(graph)

    # Draw nodes with labels
    nx.draw(graph, pos, with_labels=True, node_color='skyblue', node_size=3000, font_size=10, font_weight='bold', edge_color='gray')

    # Draw edge labels (relations)
    edge_labels = nx.get_edge_attributes(graph, 'relation')
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    # Show the plot
    plt.title("Knowledge Graph Visualization")
    plt.show()

# Example usage
# article_text = "BikeBox is a secure bicycle storage."
# llm_output_text = "BikeBox is not a secure bicycle storage."

article_text = "Norway (Norwegian Norge (Bokmål) or Noreg (Nynorsk); North Sami Norga, South Sami Nöörje, Lulesam Vuodna, Kven Norja), officially the Kingdom of Norway or Kongeriket Norge/Noreg, is a country in Northern Europe on the Scandinavian peninsula. In addition to the mainland, the Kingdom of Norway includes the archipelago of Svalbard and the island of Jan Mayen. The capital and most populous city is Oslo. The country is located in the west of the Scandinavian Peninsula and borders Sweden to the east and Finland and Russia to the northeast. Norway is one of the largest countries in Europe in terms of area (8th), but is sparsely populated with only 5,550,203 inhabitants (as of January 1, 2024). The majority of the population lives in the south of the country. As a result of the agreement concluded between Sweden and Denmark as part of the Peace of Kiel, Norway transitioned from the Union of Denmark-Norway to a union with Sweden in 1814. On May 17, 1814, Norway received its own constitution. Norway finally gained its current independence when the union with Sweden was dissolved in 1905.[6] Norway's form of government is a parliamentary monarchy."
llm_output_text = "BikeBox is a secure bicycle storage."

# Extract knowledge graphs
article_kg = extract_kg_from_article(article_text)
llm_kg = extract_kg_from_article(llm_output_text)

# print edges and nodes
print ("Article KG:")
print(article_kg.edges())
print(article_kg.nodes())

# Visualize the knowledge graph
visualize_kg(article_kg)

print ("LLM KG:")
print(llm_kg.nodes())

# Generate sentences from KGs
article_sentences = generate_sentences_from_kg(article_kg)
llm_sentences = generate_sentences_from_kg(llm_kg)

print("\nSentences from Article KG:")
for sentence in article_sentences:
    print(sentence)

print("\nSentences from LLM KG:")
for sentence in llm_sentences:
    print(sentence)

label, confidence = check_contradiction(article_text, llm_output_text)

print(f"\nLabel: {label}, Confidence: {confidence}")
