# LangChain Output Parser

- The models which by default cannot provide structured output, we need Output Parsers for them.
- Output Parsers help convert raw LLM response into structured formats like JSON, CSV, Pydantic models and more.
- They ensure consistency and validation.
- You can use output parsers with both "can" and "cannot" models
- Different Types of Output Parsers are there. Here are the most common 4
  - `StrOutputParser`
  - `JSONOutputParser`
  - `StructuredOutputParser`
  - `PydanticOutputParser`

## StrOutputParser

- This is the most simple output parser.
- Used to parse the output of LLM and return in plain string
- Mainly, when we use an LLM, it returns the output in a key called `content`. That can be fetched directly using output parser.

### Why not just use result.content instead of StrOutputParser

**Without using parser**

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate

llm = HuggingFaceEndpoint(
    repo_id='',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)


prompt_template_1 = PromptTemplate(
    template="""
    Generate a 2 liner joke on {topic}
    """,
    input_variables=['topic']
)

prompt_template_2 = PromptTemplate(
    template="""
    Generate the explanation for the below joke which was created on the topic {topic}
    Joke:
    {joke}
    """,
    input_variables=['joke', 'topic']
)

joke_chain = prompt_template_1 | model

joke_response = joke_chain.invoke({
    'topic': 'AI'
})

joke = joke_response.content

explanation_chain = prompt_template_2 | model

explanation = explanation_chain.invoke({
    'topic': 'AI',
    'joke': joke
})

print(explanation.content)
```

**Using parser**

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

llm = HuggingFaceEndpoint(
    repo_id='',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

prompt_template_1 = PromptTemplate(
    template="""
    Generate a 2 liner joke on {topic}
    """,
    input_variables=['topic']
)

# The reason of variable name is {text} because StrOutputParser object hold the LLM output value in the property called text.
# If we give any other value the pipeline will fail
prompt_template_2 = PromptTemplate(
    template="""
    Generate the explanation for the below joke {text}
    """,
    input_variables=['text']
)
# Also in the prompt_template_2, if there is any other input is required, then also we have to break the chain and build it separately.
# Something like below
# prompt_template_2 = PromptTemplate(
#     template="""
#     Generate the explanation for the below joke which is on the topic {topic}
#     {text}
#     """,
#     input_variables=['topic','text']
# )

output_parser = StrOutputParser()

chain = prompt_template_1 | model | output_parser | prompt_template_2 | model | output_parser

explanation = chain.invoke({
    'topic': 'AI'
})

print(explanation.content)
```

## JSONOutputParser

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

llm = HuggingFaceEndpoint(
    repo_id='',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

prompt_template = PromptTemplate(
    template="""
    Generate a 2 liner joke
    {instruction_about_format}
    """,
    input_variables=['topic'],
    partial_variables={'instruction_about_format': parser.get_format_instructions()}
)

# response = prompt_template.invoke({}) # you have to pass an empty dict, even if there is no put required

# print(response.text)

chain = prompt_template | model | parser

joke = chain.invoke({})

print(joke)
```

## StructuredOutputParser

- JSONOutputParser does not enforce any schema.
- StructuredOutputParser helps extract **structured JSON** from response based on predefined field schema.

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser # WHY? because it is not very much reusable
from langchain.output_parsers import ResponseSchema # This help in creating JSON Schema

llm = HuggingFaceEndpoint(
    repo_id='',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

# Need to understand a little about JSON Schema
schema = [
    ResponseSchema(name='fact_1', description='Fact 1 topic description'),
    ResponseSchema(name='fact_2', description='Fact 2 topic description'),
    ResponseSchema(name='fact_3', description='Fact 3 topic description')
]

parser = StructuredOutputParser.from_response_schemas(schema)

prompt_template = PromptTemplate(
    template="""
    Generate 3 facts about the topic {topic}
    {instruction_about_format}
    """,
    input_variables=['topic'],
    partial_variables={'instruction_about_format': parser.get_format_instructions()}
)

# response = prompt_template.invoke({'topic': 'AI'}) # you have to pass an empty dict, even if there is no put required

# print(response.text)

chain = prompt_template | model | parser

facts = chain.invoke({})

print(facts)
```

## PydanticOutputParser

- Issue with StructuredOutputParser is, you cannot enforce data validation

```python
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import PydanticOutputParser # WHY? because it is frequently used
from pydantic import BaseModel, Field

llm = HuggingFaceEndpoint(
    repo_id='',
    task='text-generation'
)

model = ChatHuggingFace(llm=llm)

class Person(BaseModel):
    name: str = Field(description='Name of the person')
    age: int = Field(gt=18, description='Age of the person')
    city: str = Field(description='City in which the person is living')

parser = PydanticOutputParser(pydantic_object=Person)

prompt_template = PromptTemplate(
    template="""
    Generate the details about a fictional person of {country}
    {instruction_about_format}
    """,
    input_variables=['country'],
    partial_variables={'instruction_about_format': parser.get_format_instructions()}
)

# response = prompt_template.invoke({'country': 'India'})

# print(response.text)

chain = prompt_template | model | parser

facts = chain.invoke({})

print(facts)
```
