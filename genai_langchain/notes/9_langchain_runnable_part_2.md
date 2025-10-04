# LangChain Runnable - Part 2

## Types of Runnable

In LangChain, we have 2 types of Runnables

1. **Task Specific Runnable**: These are the runnables which are core LangChain components. E.g. PromptTemplate, Retrieve (FakeModelComponent class in the Fake Runnable Example) etc
2. **Runnable Primitives**: These help other task specific runnables in being connected. Kinda orchestrator runnables. E.g. RunnableSequence, RunnableParallel, RunnableLambda, RunnableBranch (RunnableConnector class in the Fake Runnable Example)etc

## Runnable Primitives

### RunnableSequence

- We can connect n runnables in sequence using this primitive

```python
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()


llm = HuggingFaceEndpoint(repo_id='', task='text-generation')
model = ChatHuggingFace(llm=llm)

prompt_template = PromptTemplate(
    template = """
    What is the capital of {country}
    """,
    input_variables = ['country']
)

output_parser = StrOutputParser()

chain = RunnableSequence(prompt_template, model, output_parser)

print(chain.invoke())
```

### RunnableParallel

- Helps you in execute 2 runnables in parallel
- Each runnable receives the same input and process it independently and produce a dictionary of output

```python
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence, RunnableParallel
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace


load_dotenv()


llm = HuggingFaceEndpoint(repo_id='', task='text-generation')
model = ChatHuggingFace(llm=llm)

prompt_template_1 = PromptTemplate(
    template = """
    What is the capital of {country}
    """,
    input_variables = ['country']
)

prompt_template_2 = PromptTemplate(
    template = """
    What is the capital of {state}
    """,
    input_variables = ['state']
)


output_parser = StrOutputParser()

chain = RunnableParallel({
    "country_capital": RunnableSequence(prompt_template_1, model, output_parser),
    "state_capital": RunnableSequence(prompt_template_2, model, output_parser)
})

print(chain.invoke({
    "country": 'India',
    "state": 'Odisha'
}))
```

### RunnablePassthrough

- Whatever input it gets, returns the exact same input

```python
prompt_creation_chain = RunnableParallel(
    {
        'context': retriever | RunnableLambda(lambda docs: '\n\n'.join( [ doc.page_content for doc in docs ] )),
        'query': RunnablePassthrough()
    }
)
```

### RunnableLambda

- It can convert any python function into a runnable.
- The benefit of this is, we can use that function in with other runnable components

```python
from langchain_core.runnables import RunnablePassthrough, RunnableParallel, RunnableLambda


chain = RunnableParallel({
    'num': RunnablePassthrough(),
    'square': RunnableLambda(lambda num: num**2),
    'cube': RunnableLambda(lambda num: num**3)
})

print(chain.invoke(10))
```

    {'num': 10, 'square': 100, 'cube': 1000}

### RunnableBranch

- If Else of LangChain universe

```python
from langchain_core.runnables import (
    RunnablePassthrough,
    RunnableParallel,
    RunnableLambda,
    RunnableBranch
)

branch = RunnableBranch(
    (lambda num: num%2==0, RunnableLambda(lambda num: num**2)),
    RunnableLambda(lambda num: num**3)
)

chain = RunnableParallel({
    'num': RunnablePassthrough(),
    'square or cube': branch
})

print(chain.invoke(10))
```

    {'num': 10, 'square or cube': 100}
