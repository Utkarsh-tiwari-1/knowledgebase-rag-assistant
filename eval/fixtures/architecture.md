# Architecture notes

The assistant uses ChromaDB as its persistent vector database. Uploaded documents are split
with LangChain's recursive character text splitter and embedded with the configured OpenAI
embedding model. Retrieved excerpts are sent to the chat model with instructions to answer
only from the supplied document context.
