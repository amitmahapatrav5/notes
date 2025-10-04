# LangChain Models



## Need of LangChain Model Component

- We have various LLM Providers(OpenAI, Google, Anthropic etc), and interaction with each provider's model via api is different.
- Langchain's Model component provides a common interface using which we can connect with any of these LLM Providers.

## Types of LangChain Model

1. Language Model (Text -> Text)
   - LLM
   - Chat Model
2. Embedding Model (Text -> Vector)

## Language Model (LLM vs Chat Model)

### LLM

- LLM Models are general Purpose models. You can use them for _Text Generation_, _Summarization_, _Code Generation_ etc
- These Models take raw text as input and raw text as output
- Legacy Models. Not used anymore.
- **How these are trained:** General text Corpora(books and wikipedia data etc)
- **Memory & Context:** No built-in Memory Support
- **Role Awareness:** No understanding of roles.
- **Example Models:** GPT-3, Llama-2.7B, Mistral-7B etc

### Chat Models

- Specialized for Conversational Task
- Takes a sequence of Messages as input and Chat Messages(this is not plain text) as output.
- Newer Model
- **How these are trained:** After Base Models(LLMs) are prepared, they are fine-tuned on _Chat Dataset_(like dialogues, conversations etc)
- **Memory & Context:** Supports Structured Conversation History
- **Role Awareness:** Understands 'system', 'user' and 'assistant' roles.
- **Example Models:** GPT-3.5-turbo, GPT-4, Llama-2-Chat, Mistral-Instruct etc

## Implementation - LLM Models

```python
from langchain_openai import OpenAI
from dotenv import load_dotenv


load_dotenv()


llm = OpenAI(model='gpt-3.5-turbo-instruct')
# NOTE: We are not passing API key explicitly.
# Because the constructor will automatically fetch the API key from environment.
# Provided the key is stored against a specific key name
# In this case that is OPENAI_API_KEY

result = llm.invoke('What is the capital of India?') # This is a raw text

print(result)
```

**temperature param: When temperature=0, then for the given input, LLM is going to generate the same output all the time**

## Implementation - OpenAI - Chat Model

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv


load_dotenv()

# OPENAI_API_KEY
model = ChatOpenAI(model='gpt-4', max_completion_tokens=10)

result = model.invoke('What is the capital of India?')

print(result) # This is not a plane text
print( result.content )
```

## Implementation - HuggingFace - API - Chat Model

```python
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv


load_dotenv()

# HUGGINGFACEHUB_API_TOKEN
llm = HuggingFaceEndpoint(
    repo_id='',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

result = model.invoke('What is the capital of India?')

print(result.content)
```

## Implementation - HuggingFace - Local - Chat Model

```python
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv


load_dotenv()


llm = HuggingFacePipeline.from_model_id(
    repo_id='',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

result = model.invoke('What is the capital of India?')

print(result.content)
```

## Implementation - OpenAI - Embedding Model

```python
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv


load_dotenv()


embedding_model = OpenAIEmbeddings(model='')

docs = [
    Document(
        page_content="Sachin Tendulkar, often called the God of Cricket, holds the record for the most runs in both Test and ODI formats and is the only player to score 100 international centuries.",
        metadata={
            'sport': 'cricket'
        }
    ),
    Document(
        page_content="Cristiano Ronaldo is widely regarded as one of the greatest footballers ever, with over 790 career goals and multiple Ballon d'Or awards, showcasing his dominance across clubs and countries.",
        metadata={
            'sport': 'football'
        }
    )
]
embeddings = embedding_model.embed_documents([ doc.page_content for doc in docs ], chunk_size=500)

embedding_model.embed_query(docs[0].page_content)

print(len(embeddings))
for embedding in embeddings:
    print(embedding, end='\n\n')
```

## Implementation - HuggingFace - API - Embedding Model

## Implementation - HuggingFace - Local - Embedding Model

```python
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv


load_dotenv()


embedding_model = HuggingFaceEmbeddings(model='')

docs = [
    Document(
        page_content="Sachin Tendulkar, often called the God of Cricket, holds the record for the most runs in both Test and ODI formats and is the only player to score 100 international centuries.",
        metadata={
            'sport': 'cricket'
        }
    ),
    Document(
        page_content="Cristiano Ronaldo is widely regarded as one of the greatest footballers ever, with over 790 career goals and multiple Ballon d'Or awards, showcasing his dominance across clubs and countries.",
        metadata={
            'sport': 'football'
        }
    )
]
embeddings = embedding_model.embed_documents([ doc.page_content for doc in docs ], chunk_size=500)

embedding_model.embed_query(docs[0].page_content)

print(len(embeddings))
for embedding in embeddings:
    print(embedding, end='\n\n')
```
