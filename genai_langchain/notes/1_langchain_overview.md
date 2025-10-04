# LangChain Overview



## Why do we need LangChain

1. Building a GenAI based application requires lots of component(give the Document Chatbot example here). Writing code from scratch for this is very difficult and cumbersome, because there are lots of moving parts(Document Loader, Splitter, Embedding Model and LLM). Langchain framework provides all the components which one might need to build a GenAI based application. Only thing you have to do is configure these components.
2. Running an instance of LLM in server is a challenging and costly thing. So, companies which has created their LLM, has built an API around it and consumer can use the api to interact with LLM. But if LLM changes, we need to rewrite the code again for the component which is interacting with the LLM. Langchain provides a unified way using which, even though, we have a need to change the LLM provider, we need to make very minimal change in the code.

## Benefits of LangChain

- Concept of Chain
- Model Agnostic Development
- Complete Ecosystem (various types of Document Loader, Splitter, Vector DB Wrapper, LLM Wrapper etc)
- Memory and State Handling (Langchain provides the concept of memory)

## Alternatives of Langchain

- LlamaIndex
- Haystack
