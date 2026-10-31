# Retrieval-Augmented Generation (RAG): Learning Guide

A documentation-first introduction to Retrieval-Augmented Generation (RAG). This repository explains the main stages of a RAG workflow and collects learning resources for exploring the topic further.

> This repository is a guide, not a packaged or production-ready RAG application. The code below is an illustrative example and has not been verified as a runnable project in this repository.

## What you’ll find

- **[The guide](GUİDE.md)** — concepts, workflow, and implementation notes.
- **[Sources](SOURCE.md)** — references and further reading.
- **[Changelog](CHANGELOG.md)** — recorded documentation updates.

## RAG in brief

A typical RAG workflow retrieves relevant material and provides it to a language model as context when answering a question:

1. **Ingest:** Load documents and split them into passages.
2. **Index:** Convert passages into embeddings and store them in a searchable index.
3. **Retrieve:** Find passages relevant to a user's question.
4. **Generate:** Give the retrieved passages and question to a language model to produce an answer.
5. **Evaluate:** Check retrieval quality, answer accuracy, and whether cited passages support the answer.

Retrieval can help ground an answer in selected material, but it does not guarantee correctness or prevent hallucinations. The quality of results depends on the data, retrieval method, prompt, model, and evaluation process.

## Illustrative Python example

The following sketch shows the shape of a basic RAG pipeline. It is a starting point for learning, not a tested script included with this repository. Library APIs may change; consult their official documentation before running or adapting it.

```python
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load and split a PDF into smaller passages.
documents = PyPDFLoader("my_document.pdf").load()
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
chunks = splitter.split_documents(documents)

# Embed passages and build an in-memory FAISS index.
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
vectorstore = FAISS.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Configure a chat model with your provider credentials.
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="YOUR_GROQ_API_KEY",
)
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,
)

result = qa.invoke({"query": "What is the main idea?"})
print(result["result"])
```

This example requires compatible versions of LangChain integrations, a working Groq API key, and a local PDF at the specified path. No dependency file, sample PDF, or runnable application is currently provided in this repository.

## Further reading

Start with the [guide](GUİDE.md), then use the [source list](SOURCE.md) to explore primary documentation, papers, and tutorials. Verify current APIs and model availability against the projects' official documentation.

## Contributing

Corrections to the guide and suggestions for useful, well-sourced references are welcome. Please include the relevant source and explain what should be updated when opening an issue or pull request.

## License

See [`LICENSE (MIT License).md`](LICENSE%20%28MIT%20License%29.md) for the license file currently included in this repository.
