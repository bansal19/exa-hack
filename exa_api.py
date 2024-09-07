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
    system_prompt = "You are a helpful assistant. For the new up-to-date information provided to you, find what the old fact used to be. Return the new fact and the old fact in a CSV."
    response = open_ai_client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"This is the new fact. Find out what the old fact used to be: {new_info}"}
        ],
        max_tokens=200
    )
    summary = response.choices[0].message.content.strip()
    return summary

# Main function to build the CSV
def build_csv(filename, topics):
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["Fact", "Old Information", "New Information"])

        for topic in topics:
            results = search_info(topic)
            if results:
                new_info = results[1].text if len(results) > 1 else "No new information found"
                row = [topic, get_old_and_new_info(new_info)]
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
