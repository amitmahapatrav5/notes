# LangChain Runnable - Part 1

## Issue with LangChain before runnable

- ChatGPT Released on 2022 November and OpenAI exposed it's API to public, so they can make GenAI based app.
- This time LangChain team thought of making a framework which will help developers build GenAI based app(like ChatBot, PDF Reader etc)
- LangChain team observed that other companies like Google, Anthropic etc are also coming up with their own llm and exposing their respective API.
- So LangChain team built **Vendor Based Classes** using which developers can build GenAI based app, and with change in provider, they need to make minimal code change.
- Then LangChain realized that, for making a GenAI based app, other components like DocumentLoader, TextSplitter, Embedding Model, Retriever, Output Parser, Vector DB etc are also required. So they built Specific Components for that.
- Then LangChain team realized that, in every application, some things are so common. Like in every application, you are going to build prompt, using prompt template and submit that prompt to LLM. This task was done by developers. So they thought of making some built-in functions which will take the components and execute them in the intended flow. They named these built-in function as **chain**. And the most common chain was LLMChain. This will take the LLM and Prompt and provide the output. Similarly for RAG applications, they made **RetrieverQAChain**. Similarly they find out other commonly used pattern and started making chains for that.
- Eventually LangChain team made a lot of chains and that raised 2 major issues.
  1. this made the codebase too big which was very difficult to maintain
  2. the learning curve became very steep for newer developer.
- Their initial goal was to make individual components and the developers can use those components to make any type of workflow they need. But that didn't happen and they are now in trouble because of som many chains which made the codebase is too large and learning curve is so steep.
- Question, why they need to make so many chains? Because the individual components are not standardized. That means, to interact with LLM Component, you have to call `predict()` method, wheres to interact with prompt component, you have to use `format()` method. Similarly to deal with different components, different methods are there, which are difficult to plug. So when they wanted to plug 2 components, they have to write manual functions and write code to pass result of one component output to the next component.
- Then they realized that, if these components were standardized, then they don't have to write custom code to plug 2 components.
- Then that became possible with something called **Runnable**.

## What is a Runnable

- Runnable is basically a unit of work, which basically takes an input, process it and returns and output.
- They have a common interface(common set of methods like invoke, batch, stream etc)
- And because the Runnables follow a common interface, you can connect them very easily, without writing any glue code.
- And by connecting these runnables when you make a workflow, that workflow itself is a runnable. What benefit do we get with that? Say we have 2 workflows w1 (r1->r2->r3) and w2 (r4->r5). Because these workflows are runnable, we can connect them together as well, w1->w2. In place of w1 and w2, it can be any complex workflow, not just sequential ones.
- In a layman term, A runnable is like a Lego Block.

## Implementation - Before Runnable

```python
from typing import List, Dict


class FakeModelComponent:
    def __init__(self, model: str):
        self.model = model
    def predict(self, prompt):
        return 'something'

# Test Code
model = FakeModelComponent(model='gpt-4')
prediction = model.predict('Tell me something')
print(prediction)

class FakePromptTemplateComponent:
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables
    def format(self, input_variables: Dict[str, str]):
        return self.template.format(**input_variables)

# Test Code
prompt_template = FakePromptTemplateComponent(
    template= """
        Please tell me if this statement is true or not.
        {capital} is the capital of {country}
    """,
    input_variables= ['capital', 'country']
)

prompt = prompt_template.format(input_variables={
    'capital': 'Delhi',
    'country': 'India'
})

print(prompt)

class RunnableConnector:
    def __init__(self, model: FakeModelComponent, prompt_template: FakePromptTemplateComponent):
        self.model = model
        self.prompt_template = prompt_template

    def run(self, input_variables):
        # manual code
        prompt = self.prompt_template.format(input_variables=input_variables)
        prediction = self.model.predict(prompt)
        return prediction


chain = RunnableConnector(
    FakeModelComponent(model='gpt-4'),
    FakePromptTemplateComponent(
        template= """
            Please tell me if this statement is true or not.
            {capital} is the capital of {country}
            """,
        input_variables= ['capital', 'country']
    )
)

result = chain.run(
    input_variables={
        'capital': 'Delhi',
        'country': 'India'
    }
)

print(result)
```

    something

            Please tell me if this statement is true or not.
            Delhi is the capital of India

    something

## Implementation - After Runnable

```python
from abc import ABC, abstractmethod
from typing import List, Dict

class Runnable(ABC):
    @abstractmethod
    def invoke(self, *args, **kwargs):
        ...


class FakeModelComponent(Runnable):
    def __init__(self, model: str):
        self.model = model
    def invoke(self, prompt):
        return {'response': 'something'}

# # Test Code
# model = FakeModelComponent(model='gpt-4')
# prediction = model.invoke('Tell me something')
# print(prediction.get('response'))

class FakePromptTemplateComponent(Runnable):
    def __init__(self, template: str, input_variables: List[str]):
        self.template = template
        self.input_variables = input_variables

    def invoke(self, input_variables: Dict[str, str]):
        return self.template.format(**input_variables)

# # Test Code
# prompt_template = FakePromptTemplateComponent(
#     template= """
#         Please tell me if this statement is true or not.
#         {capital} is the capital of {country}
#     """,
#     input_variables= ['capital', 'country']
# )

# prompt = prompt_template.invoke(input_variables={
#     'capital': 'Delhi',
#     'country': 'India'
# })

# print(prompt)

class RunnableConnector(Runnable):
    def __init__(self, components: List[Runnable]):
        self.components = components

    def invoke(self, args):
        result = args
        for component in self.components:
            result = component.invoke(result)
        return result


# Chain 1
prompt_1 = FakePromptTemplateComponent(
    template= """
        Please tell me if this statement is true or not.
        {capital} is the capital of {country}
        """,
    input_variables= ['capital', 'country']
)
model_1 = FakeModelComponent(model='gpt-4')

chain_1 = RunnableConnector([prompt_1, model_1])


# Chain 2
prompt_2 = FakePromptTemplateComponent(
    template="""
    Please suggest me if I should visit {response}.
    """,
    input_variables=['response']
)
model_2 = FakeModelComponent(model='gpt-oss')

chain_2 = RunnableConnector([prompt_2, model_2])

chain = RunnableConnector([chain_1, chain_2])

result = chain.invoke({
    'capital': 'Delhi',
    'country': 'India'
})

print(result.get('response'))
```

    something

```python

```
