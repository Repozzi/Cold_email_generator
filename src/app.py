import streamlit as st
import os
from groq import Groq
from together import Together
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
together_api_key = os.getenv("TOGETHER_API_KEY")
pplx_api_key = os.getenv("PPLX_API_KEY")

# Initialize your clients (you'll need to handle API keys securely)
groq_client = Groq(api_key=groq_api_key)
together_client = Together(api_key=together_api_key)
openai_client = OpenAI(api_key=pplx_api_key, base_url="https://api.perplexity.ai")


llama_70B = "llama-3.1-70b-versatile"
llama_405B = "llama-3.1-405b-reasoning"


def main():
    st.title("Cold Email Generator")

    # Replace get_user_input() with Streamlit input
    target = st.text_input("Please enter the industry or company you want to target:")

    if st.button("Generate Cold Email"):
        if target:
            with st.spinner("Generating queries..."):
                generated_queries = query_agent(target)

            st.subheader("Generated Queries:")
            for query in generated_queries:
                st.write(query)

            with st.spinner("Performing web search..."):
                search_results = []
                for query in generated_queries:
                    result = web_search_agent(query)
                    search_results.append(result)

            st.subheader("Search Results:")
            for i, result in enumerate(search_results):
                st.write(f"Query {i+1} results:")
                st.write(result)

            with st.spinner("Generating cold emails..."):
                cold_email_405B = cold_email_agent_405B(target, search_results)
                cold_email_70B = cold_email_agent_70B(target, search_results)

            st.subheader("Generated Cold Emails:")
            st.write("405B Model Email:")
            st.text(cold_email_405B)
            st.write("70B Model Email:")
            st.text(cold_email_70B)

            # Option to download emails
            st.download_button(
                label="Download 405B Model Email",
                data=cold_email_405B,
                file_name="cold_email_405B.txt",
                mime="text/plain",
            )
            st.download_button(
                label="Download 70B Model Email",
                data=cold_email_70B,
                file_name="cold_email_70B.txt",
                mime="text/plain",
            )
        else:
            st.warning("Please enter a target industry or company.")


def query_agent(target):
    response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": """You are an AI assistant that writes concise search queries for market research.
                Create 4 short search queries based on the given target industry or company.
                Query 01: biggest pain points faced by this avatar
                Query 02: biggest companies in this industry
                Query 03: how companies in this industry get clients
                Query 04: where to find companeis in this industry online
                IMPORTANT: Respond with only the queries, one per line.""",
            },
            {
                "role": "user",
                "content": f"Here's the industry / company to perform market research on: #### {target} ####",
            },
        ],
        max_tokens=150,
        temperature=0.7,
        top_p=1,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>"],
    )
    queries = response.choices[0].message.content.split("\n")
    return [query.strip() for query in queries if query.strip()]


def web_search_agent(query):
    response = openai_client.chat.completions.create(
        model="llama-3-sonar-large-32k-online",
        messages=[
            {
                "role": "system",
                "content": "You are a web search assistant. Provide a concise summary of the search results.",
            },
            {"role": "user", "content": f"Search the web for: {query}"},
        ],
    )
    return response.choices[0].message.content


def cold_email_agent_70B(target, search_results):
    combined_results = "\n".join(search_results)

    chat_completion = groq_client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an expert cold email writer.
                    Your task is to write concise and personalized cold emails based on the Market Research given to you.
                    Make sure to utilize all 4 areas of the research I(pain points, companies, clients, and online sources)
                    Focus on describing what the target avatar will get, add an appealing guarantee.
                    Keep the email concise and use plain English.
                    #### DO NOT OUTPUT OTHER INFORMATION EXCEPT COLD EMAIL ITSELF !!! ONLY THE COLD EMAIL ITSELF! ####.
                    """,
            },
            {
                "role": "user",
                "content": f"Here is the target avatar: {target} \n Here is the market research: #### {combined_results} #### ONLY OUTPUT THE EMAIL ITSELF. NO OTHER TEXT!!",
            },
        ],
        model=llama_70B,
    )

    return chat_completion.choices[0].message.content


def cold_email_agent_405B(target, search_results):
    # Combine all search results into a single string
    combined_results = "\n".join(search_results)

    response = together_client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo",
        messages=[
            {
                "role": "system",
                "content": """You are an expert cold email writer.
                Your task is to write concise and personalized cold emails based on the Market Research given to you.
                Make sure to utilize all 4 areas of the research I(pain points, companies, clients, and online sources)
                Focus on describing what the target avatar will get, add an appealing guarantee.
                Keep the email concise and use plain English.
                #### DO NOT OUTPUT OTHER INFORMATION EXCEPT COLD EMAIL ITSELF !!! ONLY THE COLD EMAIL ITSELF! ####.
                """,
            },
            {
                "role": "user",
                "content": f"Here is the target avatar: {target} \n Here is the market research: #### {combined_results} #### ONLY OUTPUT THE EMAIL ITSELF. NO OTHER TEXT!!",
            },
        ],
        max_tokens=500,
        temperature=0.1,
        top_p=1,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>"],
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    main()
