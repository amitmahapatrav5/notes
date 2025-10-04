# LangChain Chain

- We can make Sequential, Parallel, Conditional chains in LangChain

## LCEL

- Langchain team observed that RunnableSequence is used very frequently, independently as well as in other runnables.
- So they thought of creating a new syntax for creating the same chain which is more readable and declarative.
- E.g. prompt | model | output_parser
- This is called **LangChain Expression Language**, or **LCEL**

## Sequential Chain

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_core.output_parsers import StrOutputParser

prompt_template = PromptTemplate(
    template="""
    {capital} is the capital of {country}
    True/False
    """,
    input_variables=['capital', 'country']
)

output_parser = StrOutputParser()

fake_model = RunnableLambda( lambda prompt: str(True) )


chain = prompt_template | fake_model | output_parser

result = chain.invoke({
    'capital': 'Delhi',
    'country': 'India'
})

print(result)
chain.get_graph().print_ascii()
```

    True
         +-------------+
         | PromptInput |
         +-------------+
                *
                *
                *
        +----------------+
        | PromptTemplate |
        +----------------+
                *
                *
                *
            +--------+
            | Lambda |
            +--------+
                *
                *
                *
       +-----------------+
       | StrOutputParser |
       +-----------------+
                *
                *
                *
    +-----------------------+
    | StrOutputParserOutput |
    +-----------------------+

```python
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

prompt_template_1 = PromptTemplate(
    template="""
    Generate 5 line note on the topic {topic}
    """,
    input_variables=['num']
)

prompt_template_2 = PromptTemplate(
    template="""
    Generate 5 multiple choice quiz question on the topic {topic}
    """,
    input_variables=['num']
)

prompt_template_3 = PromptTemplate(
    template="""
    Please merge the Given Note and Quiz below
    Note:
    {note}
    Quiz:
    {quiz}
    """,
    input_variables=['note', 'quiz']
)

model_1 = RunnableLambda( lambda prompt: 'note' )
model_2 = RunnableLambda( lambda prompt: 'quiz' )
model_3 = RunnableLambda( lambda prompt: 'Note \nQuiz')

output_parser = StrOutputParser()

chain = RunnableParallel({
    'note': prompt_template_1 | model_1 | output_parser,
    'quiz': prompt_template_1 | model_2 | output_parser
}) | prompt_template_3 | model_3 | output_parser

result = chain.invoke({
    'topic': 'Decoder only transformer models are used in building LLMs.'
})

print(result)

chain.get_graph().print_ascii()
```

    Note
    Quiz
              +--------------------------+
              | Parallel<note,quiz>Input |
              +--------------------------+
                    ***           ***
                  **                 **
                **                     **
    +----------------+            +----------------+
    | PromptTemplate |            | PromptTemplate |
    +----------------+            +----------------+
              *                           *
              *                           *
              *                           *
        +--------+                    +--------+
        | Lambda |                    | Lambda |
        +--------+                    +--------+
              *                           *
              *                           *
              *                           *
    +-----------------+          +-----------------+
    | StrOutputParser |          | StrOutputParser |
    +-----------------+          +-----------------+
                    ***           ***
                       **       **
                         **   **
              +---------------------------+
              | Parallel<note,quiz>Output |
              +---------------------------+
                            *
                            *
                            *
                   +----------------+
                   | PromptTemplate |
                   +----------------+
                            *
                            *
                            *
                       +--------+
                       | Lambda |
                       +--------+
                            *
                            *
                            *
                  +-----------------+
                  | StrOutputParser |
                  +-----------------+
                            *
                            *
                            *
                +-----------------------+
                | StrOutputParserOutput |
                +-----------------------+

## Conditional Chain

```python
from random import choice

from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableBranch
from langchain_core.output_parsers import StrOutputParser


prompt_template_1 = PromptTemplate(
    template="""
    Provide the sentiment of the below customer review as 'positive' or 'negative'
    {review}
    """
)

prompt_template_2 = PromptTemplate(
    template="""
    Suggest some products for the below customer based on his review
    Customer Review: {review}
    """,
    input_variables=['review']
)

prompt_template_3 = PromptTemplate(
    template="""
    Schedule a dedicated customer care agent call for the customer based on the review
    Customer Review: {review}
    """,
    input_variables=['review']
)

model_1 = RunnableLambda( lambda prompt: choice(['positive', 'negative']) )
model_2 = RunnableLambda( lambda prompt: 'Product Suggestion' )
model_3 = RunnableLambda( lambda prompt: 'Scheduled customer care call' )

output_parser = StrOutputParser()

sentiment_chain = prompt_template_1 | model_1 | output_parser

branch_chain = RunnableBranch(
    (RunnableLambda( lambda sentiment: sentiment == 'positive' ), prompt_template_2 | model_2 | output_parser ),
    (RunnableLambda( lambda sentiment: sentiment == 'negative' ), prompt_template_3 | model_3 | output_parser ),
    RunnableLambda(lambda prompt: 'The sentiment of the review was neutral')
)

chain = sentiment_chain | branch_chain

result = chain.invoke({
    'review': 'Dummy Review'
})

print(result)

chain.get_graph().print_ascii()
```

    Product Suggestion
      +-------------+
      | PromptInput |
      +-------------+
              *
              *
              *
    +----------------+
    | PromptTemplate |
    +----------------+
              *
              *
              *
        +--------+
        | Lambda |
        +--------+
              *
              *
              *
    +-----------------+
    | StrOutputParser |
    +-----------------+
              *
              *
              *
        +--------+
        | Branch |
        +--------+
              *
              *
              *
      +--------------+
      | BranchOutput |
      +--------------+
