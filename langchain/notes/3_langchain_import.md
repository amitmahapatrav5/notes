
# LangChain Component Imports



## 5 Types of Imports

```python
# from langchain_core.<component> import something
from langchain_core.documents import Document

# from langchain.<component> import something
from langchain.chains import LLMChain

# from langchain_community.<component> import something
from langchain_community.document_loaders import TextLoader

# from langchain_<vendor> import something
from langchain_chroma import Chroma

# from langchain_text_splitters import something
from langchain_text_splitters import RecursiveCharacterTextSplitter
```

## Embeddings

```python
# Interface
from langchain_core.embeddings import Embeddings

# Vendor implementations
from langchain_openai import OpenAIEmbeddings # OpenAI
from langchain_aws import BedrockEmbeddings # AWS Bedrock
```

## Document Loader

```python
# Interface
from langchain_core.document_loaders import BaseLoader

# Concrete loaders
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
```

## Splitter

```python
# All text splitters are in their own package now
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_text_splitters import CharacterTextSplitter
from langchain_text_splitters import TokenTextSplitter
```

## Vector Store

```python
# Interfaces
from langchain_core.vectorstores import VectorStore
from langchain_core.vectorstores import InMemoryVectorStore

# Vendor/community implementations
from langchain_chroma import Chroma                        # Chroma (vendor package)
from langchain_community.vectorstores.faiss import FAISS   # FAISS (community)
```

## Prompt

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import MessagesPlaceholder
```

## Messages

```python
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage
from langchain_core.messages import SystemMessage
```

## Retriever

```python
# Interface
from langchain_core.retrievers import BaseRetriever

# Vector store-backed retriever
from langchain_core.vectorstores import VectorStoreRetriever
```

## Runnable

```python
# Interface
from langchain_core.runnables import Runnable

from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables import RunnableParallel
```

## Chain

```python
# Modern style: compose with LCEL (runnables + | operator) — no extra import needed.

from langchain.chains import LLMChain # Legacy
```

## Chat Model

```python
# Vendor chat models
from langchain_openai import ChatOpenAI              # OpenAI
from langchain_anthropic import ChatAnthropic        # Anthropic
```

## Output Parser

```python
from langchain_core.output_parsers import StrOutputParser
from langchain_core.output_parsers import PydanticOutputParser
```

## Memory

```python
# Interface
from langchain_core.chat_history import BaseChatMessageHistory

# Various Types of Chat History
from langchain_community.chat_message_histories import FileChatMessageHistory
from langchain_community.chat_message_histories import RedisChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
```
