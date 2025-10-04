# LangChain Prompt Template


**Types of Prompt**

1. Static Prompt (Raw text sent by the used. example: "What is the capital of India")
2. Dynamic Prompt (using `PromptTemplate(template="something Context: {context} Query:{query}", input_variables=['context', 'query'])`)

## PromptTemplate

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.prompt_values import PromptValue


prompt_template = PromptTemplate(
    template="""
        You are a helpful assistant that answers strictly based on the provided context.

        Guidelines:
        - Use only the information in the Context to answer the Query.
        - If the answer is not present in the Context, say "I don't have enough information in the provided context to answer that."
        - Be concise: 1 short paragraph, maximum 2-3 sentences.
        - Do not invent facts, do not speculate, and do not use external knowledge.
        - If multiple relevant points exist in Context, synthesize them clearly.
        - Preserve any important terminology from the Context.

        Context:
        {context}

        Query:
        {query}

        Answer:
    """,
    input_variables=["context", "query"],
)

# prompt: PromptValue
prompt: PromptValue = prompt_template.invoke({
    'context': 'Virat Kohli is a Cricketer.',
    'query': 'What sport does Virat play?'
})

print(prompt.to_string())
# print(prompt.to_messages()) # HumanMessage
```

            You are a helpful assistant that answers strictly based on the provided context.

            Guidelines:
            - Use only the information in the Context to answer the Query.
            - If the answer is not present in the Context, say "I don't have enough information in the provided context to answer that."
            - Be concise: 1 short paragraph, maximum 2-3 sentences.
            - Do not invent facts, do not speculate, and do not use external knowledge.
            - If multiple relevant points exist in Context, synthesize them clearly.
            - Preserve any important terminology from the Context.

            Context:
            Virat Kohli is a Cricketer.

            Query:
            What sport does Virat play?

            Answer:

## Why not a Python F-String

```python
# We get a built-in validation, f-string does not provide that

name = input('Enter Your Name: ')
print(f'Hello {name}')
```

    Enter Your Name:


    Hello

```python
# Prompt Templates can be stored outside the codebase in a separate file.

# Store
from langchain_core.prompts import PromptTemplate, load_prompt
from langchain_core.prompt_values import PromptValue


prompt_template = PromptTemplate(
    template="""
        You are a helpful assistant that answers strictly based on the provided context.

        Guidelines:
        - Use only the information in the Context to answer the Query.
        - If the answer is not present in the Context, say "I don't have enough information in the provided context to answer that."
        - Be concise: 1 short paragraph, maximum 2-3 sentences.
        - Do not invent facts, do not speculate, and do not use external knowledge.
        - If multiple relevant points exist in Context, synthesize them clearly.
        - Preserve any important terminology from the Context.

        Context:
        {context}

        Query:
        {query}

        Answer:
    """,
    input_variables=["context", "query"],
    validate_template=True
)

prompt_template.save('test.json')

# Load
from langchain_core.prompts import load_prompt
from langchain_core.prompt_values import PromptValue


prompt_template = load_prompt('test.json')

# prompt: PromptValue
prompt: PromptValue = prompt_template.invoke({
    'context': 'Virat Kohli is a Cricketer.',
    'query': 'What sport does Virat play?'
})

print(prompt.to_string())

```

            You are a helpful assistant that answers strictly based on the provided context.

            Guidelines:
            - Use only the information in the Context to answer the Query.
            - If the answer is not present in the Context, say "I don't have enough information in the provided context to answer that."
            - Be concise: 1 short paragraph, maximum 2-3 sentences.
            - Do not invent facts, do not speculate, and do not use external knowledge.
            - If multiple relevant points exist in Context, synthesize them clearly.
            - Preserve any important terminology from the Context.

            Context:
            Virat Kohli is a Cricketer.

            Query:
            What sport does Virat play?

            Answer:

```python
# Prompt Template is tightly coupled in Langchain ecosystem. It is a runnable
```

## LangChain Messages

### Types of LangChain Messages

1. System Message
2. Human Message
3. AI Message

```python
from typing import List

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import load_prompt
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

chat_history = [
    SystemMessage(content="You are a helpful assistant! Your name is Bob."),
]

# llm = HuggingFaceEndpoint(
#     repo_id='',
#     task='text-generation'
# )

# model = ChatHuggingFace(llm=llm)


class FakeChatModel:
    def invoke(self, chat_history: List[HumanMessage]):
        return AIMessage(content=chat_history[-1].content[::-1])


model = FakeChatModel()

while True:
    query = input('Enter Your Query: ')
    if query=='exit':
        print(chat_history)
        break
    else:
        chat_history.append(HumanMessage(content=query))
        response = model.invoke(chat_history)
        print(f'Bob: {response.content}')
        chat_history.append(response.content)
```

    Enter Your Query:  test


    Bob: tset


    Enter Your Query:  exit


    [SystemMessage(content='You are a helpful assistant! Your name is Bob.', additional_kwargs={}, response_metadata={}), HumanMessage(content='test', additional_kwargs={}, response_metadata={}), 'tset']

IMAGE

## ChatPromptTemplate

```python
from langchain_core.prompts import ChatPromptTemplate

chat_prompt_template = ChatPromptTemplate([
    ('system', 'You are a helpful {expert} expert'),
    ('human', 'Explain me this, in {context}, topic {topic}')
])

prompt = chat_prompt_template.invoke({
    "expert": 'Cricket',
    "context": 'Batting',
    "topic": "Strike Rate"
})

print(prompt.to_string())
```

    System: You are a helpful Cricket expert
    Human: Explain me this, in Batting, topic Strike Rate

## MessagePlaceholder

If the user comes to a particular chat which he had in the past and asks about something, then we need to retrieve what conversation user has had in the part in that chat history and feed that to llm before answering user's query. That's where MessagePlaceholder comes into the play.

```python
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


chat_prompt_template = ChatPromptTemplate([
    ('system', 'You are a helpful assistant'),
    MessagesPlaceholder(variable_name='chat_history'),
    ('human', '{query}')
])

chat_history = [
    HumanMessage(content='What is the status of my appointment'),
    AIMessage(content='It is scheduled on 24th September 2025'),
    HumanMessage(content='Ok')
]

prompt = chat_prompt_template.invoke({
    'chat_history': chat_history,
    'query': 'When have you schedules my appointment with the doc?'
})

print(prompt.to_string())
```

    System: You are a helpful assistant
    Human: What is the status of my appointment
    AI: It is scheduled on 24th September 2025
    Human: Ok
    Human: When have you schedules my appointment with the doc?

```python

```
