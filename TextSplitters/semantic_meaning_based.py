from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings

embedding=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

splitter=SemanticChunker(
    embedding,breakpoint_threshold_type='standard_deviation',
    breakpoint_threshold_amount=1
)

text="""
Artificial Intelligence is transforming modern technology. It enables machines
to learn from data, recognize patterns, and make decisions with minimal human intervention.
AI is widely used in recommendation systems, chatbots, and autonomous vehicles. Machine Learning is a subset of AI that focuses on algorithms that improve through experience.
Supervised learning, unsupervised learning, and reinforcement learning are its main categories.
These techniques are commonly applied in image recognition and natural language processing. Climate change is one of the biggest challenges facing humanity today.
Rising global temperatures, melting ice caps, and extreme weather events
are clear indicators of environmental imbalance. Governments and industries
are investing in renewable energy and sustainable practices to reduce carbon emissions.
"""

result=splitter.split_text(text)

print(result)