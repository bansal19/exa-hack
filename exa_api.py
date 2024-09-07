import csv
from exa_py import Exa
import os
from openai import OpenAI

# Initialize the Exa client
exa = Exa(api_key=os.environ["EXA_API_KEY"])
open_ai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Function to search for information and return results
def search_info(query):
    results = exa.search_and_contents(
        query,
        use_autoprompt=True,
        num_results=5,
        text={"max_characters": 1000}
    )
    return results.results

def get_old_and_new_info(new_info):
    response = open_ai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant. For the new up-to-date information provided to you, find what the old fact used to be."},
            {"role": "user", "content": f"This is the new fact. Find out what the old fact used to be: {new_info}"}
        ],
        max_tokens=200
    )
    old_fact = response.choices[0].message.content.strip()

    response_new_fact = open_ai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Summarise the following please: {new_info}"}
        ],
        max_tokens=200
    )
    summarised_new_fact = response_new_fact.choices[0].message.content.strip()

    return old_fact, summarised_new_fact

# Main function to build the CSV
def build_csv(filename, topics):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Fact", "Old Information", "New Information"])

        for topic in topics:
            results = search_info(topic)
            if results:
                new_info = results[1].text if len(results) > 1 else "No new information found"
                row = [topic, get_old_and_new_info(new_info)[0], get_old_and_new_info(new_info)[1]]
                writer.writerow(row)

# List of topics or facts to search for
topics = [
    "Updates and advancements in Politics",
    "Artificial intelligence advancements",
    "Climate change impact",
    "Developments in quantum computing",
    "Recent breakthroughs in renewable energy"
]

# Build the CSV
build_csv("llm_update_dataset.csv", topics)

print("CSV file 'llm_update_dataset.csv' has been created.")
