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
article_text = "BikeBox is a secure bicycle storage."
llm_output_text = "BikeBox is not a secure bicycle storage."

# Extract knowledge graphs
article_kg = extract_kg_from_article(article_text)
llm_kg = extract_kg_from_article(llm_output_text)

# Generate sentences from KGs
article_sentences = generate_sentences_from_kg(article_kg)
llm_sentences = generate_sentences_from_kg(llm_kg)

print("\nSentences from Article KG:")
for sentence in article_sentences:
    print(sentence)

print("\nSentences from LLM KG:")
for sentence in llm_sentences:
    print(sentence)

# Visualize the knowledge graph
# visualize_kg(llm_kg)

label, confidence = check_contradiction(article_text, llm_output_text)

print(f"\nLabel: {label}, Confidence: {confidence}")
