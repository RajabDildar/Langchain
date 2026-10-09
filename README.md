# Langchain Learning Repository

A comprehensive repository exploring different components of **LangChain** in detail. This repo serves as personal learning notes and reference material for understanding LangChain's core concepts and practical implementations.

> **Note**: This is a learning resource repository with hands-on examples, demonstrations, and experiments for mastering LangChain framework.

---

## 📚 Repository Overview

LangChain is a powerful Python framework for building applications with Large Language Models (LLMs). This repository breaks down LangChain into its fundamental components, providing practical code examples and demonstrations for each.

---

## 📁 Project Structure

### 1. **Langchain-Prompts**
Understanding and working with prompt engineering in LangChain.

- **chatPromptTemplate.py** - Demonstrates how to create and structure chat prompt templates
- **chatbot.py** - Basic chatbot implementation using prompts
- **messages.py** - Working with different message types in LangChain

**Key Concepts**: Prompt templates, message formatting, conversation structure

---

### 2. **Langchain-Models**
Exploration of different model types and integrations available in LangChain.

#### **LLMS/** - Large Language Models
Working directly with LLM integrations like OpenAI, HuggingFace, etc.

#### **Chat-Models/** - Chat-based Models
Using chat-based LLM endpoints optimized for conversational interactions.

#### **Embedding-Models/** - Embedding Models
Converting text to vector embeddings for semantic search and similarity operations.

**Files**: 
- requirements.txt - Dependencies for model implementations

**Key Concepts**: Model selection, API integration, embedding generation, tokenization

---

### 3. **Langchain-Tools**
Building and integrating custom tools that LLMs can use.

**Key Concepts**: Tool creation, tool calling, action execution, integration with agents

---

### 4. **Langchain-Outputs**
Managing and structuring outputs from LLM responses.

**Key Concepts**: Output parsing, response formatting, data extraction

---

### 5. **Langchain-Chaining**
Connecting multiple components together to build complex workflows.

**Key Concepts**: Chain composition, sequential processing, conditional logic, workflow automation

---

### 6. **Langchain-AI-Agents**
Building intelligent agents that can use tools and make decisions.

**Key Concepts**: Agent creation, ReAct pattern, decision-making, tool selection

---

### 7. **Langchain-RAG** ⭐
Retrieval-Augmented Generation implementation with various data sources and vector databases.

#### **Document Loaders**
- **pdf-loader.py** - Load and process PDF documents
- **doc-loader.py** - Load various document formats (.docx, .txt, etc.)
- **directory-loader.py** - Load all documents from a directory
- **web-loader.py** - Scrape and load content from web pages
- **retriever-wikipedia.py** - Fetch information from Wikipedia

#### **Text Splitting**
- **text-splitter-char-length.py** - Split documents by character count
- **text-splitter-char-length2.py** - Alternative character-based splitting
- **recursive-char-text-splitter.py** - Recursive splitting preserving semantic boundaries

#### **Vector Database**
- **vector-db.py** - Creating and managing vector embeddings in Chroma DB
- **chroma_db/** - Vector database storage directory

#### **Retrieval Strategies**
- **vector-retriever.py** - Basic semantic search using vectors
- **vector-retriever-mmr.py** - Maximum Marginal Relevance retrieval (diverse results)
- **vector-retriever-mqr.py** - Multi-query retrieval for better coverage

#### **Sample Data**
- **books/** - Directory for storing book files
- **cricket.txt** - Sample text file for testing
- **dl-curriculum.pdf** - Sample PDF for document processing

**Key Concepts**: Document loading, text chunking, embeddings, vector similarity, retrieval techniques

---

### 8. **YT-Chatbot-using-RAG** 🎥
Real-world implementation: A chatbot that answers questions based on YouTube video content.

**Key Concepts**: Multi-source RAG, video content processing, interactive Q&A

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip (Python package manager)

### Installation

Each subdirectory may have its own `requirements.txt` file. Install dependencies for the specific component you're working with:

```bash
# For RAG examples
cd Langchain-RAG
pip install -r requirements.txt

# For Models
cd Langchain-models
pip install -r requirements.txt
```

### Running Examples

Navigate to any directory and run the Python scripts:

```bash
python script_name.py
```

---

## 🔑 Key LangChain Concepts Covered

| Concept | Location | Purpose |
|---------|----------|---------|
| **Prompts** | Langchain-Prompts | Template-based prompt engineering |
| **Models** | Langchain-Models | LLM and embedding model integration |
| **Tools** | Langchain-Tools | Custom tool creation and calling |
| **Chains** | Langchain-Chaining | Sequential component orchestration |
| **Agents** | Langchain-AI-Agents | Autonomous decision-making |
| **RAG** | Langchain-RAG | Knowledge retrieval and augmentation |
| **Real-World App** | YT-Chatbot-using-RAG | End-to-end application example |

---

## 💡 Learning Path Recommendation

1. **Start here**: Langchain-Prompts (foundation)
2. **Then**: Langchain-Models (core building block)
3. **Next**: Langchain-Outputs (result processing)
4. **Advanced**: Langchain-Chaining (combining components)
5. **Practical**: Langchain-Tools & Agents
6. **Real-world**: Langchain-RAG & YT-Chatbot-using-RAG

---

## 📖 Resources & References

- [Official LangChain Documentation](https://docs.langchain.com/)
- [LangChain GitHub Repository](https://github.com/langchain-ai/langchain)
- [LangChain API Reference](https://api.python.langchain.com/)

---

## ✨ Notes

- This repository contains **learning examples and notes** for personal reference
- Code examples are designed to be **educational and modular**
- Each component can be studied and adapted independently
- Feel free to fork, modify, and use these examples for your learning journey!

---

## 📝 License

This repository is for educational purposes. Use the code examples as learning material for understanding LangChain concepts.

---

**Last Updated**: October 9, 2026

*Happy Learning! 🚀*
