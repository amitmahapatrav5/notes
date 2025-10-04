# LangChain Components

## Important Components of LangChain

1. Model
2. Prompts
3. Chains
4. Memory
5. Indexes
6. Agents

## Model

- We interact with LLMs via the API built by the LLM provider. Different provider has different API structure. So consumption codebase is different. **Standardization became a challenge.**

### Provider API Specific Code

```python
# Code to connect with LLM Provider 1
```

```python
# Code to connect with LLM Provider 2
```

### LangChain is used to connect with LLMs

**Ref: https://python.langchain.com/docs/integrations/chat/**

```python
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model='gpt-4')

result = model.invoke('What is the capital of India?')

print(result.content)
```

```python
from langchain_openai import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(model='claude-3-7-sonnet-20250219')

result = model.invoke('What is the capital of India?')

print(result.content)
```

### Types of Langchain Language Models

1. Language Model (Text In -> Text Out)
   - LLM
   - ChatModel
2. Embedding Model (Text In -> Vector Out)

## Prompts

- The input that we are providing to LLM.
- The response we get from LLM highly depends on our prompt. There is a field of study, Prompt Engineering.
- So Langchain team created a component using which u can use various prompting technique that can be sent to LLM.
  - PromptTemplate
  - ChatPromptTemplate
  - FewShotPromptTemplate

## Chains

- You can build pipelines using this Chains Component
- It does the heavy-lifting of getting the output from previous component and sending it to next.
- Using chains, you can make Sequential, Conditional and Parallel Chains

## Memory

- LLM API Calls are Stateless(An example of this)
- Various Types of Memory Component are there
  - ConversationalBufferMemory
  - ConversationBufferWindowMemory
  - Summarizer-Based Memory
  - Custom Memory

## Indexes

- Used to connect LLM to External Knowledge
- Made up of 4 main Components
  1. **Document Loader**
  2. **Text Splitter**
  3. **Vector Store**
  4. **Retriever**

## Agents

- LLMs have 2 USP(NLU + Text Generation) but cannot perform any action
- An AI Agent, can do what LLMs can do + Perform Actions (API Call mainly) = (Reasoning Capability + Tools Calling)
- But I won't recommend to use LangChain to build Agents. Because it's not scalable and you have to write lots of glue code. Hence we have LangGraph.
