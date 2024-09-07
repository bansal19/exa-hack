import csv
import json
from exa_py import Exa
import os
from openai import OpenAI
from datetime import datetime, timedelta
import pandas as pd

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

def search_and_validate(query):
	results = exa.search_and_contents(
		query,
		use_autoprompt=True,
		num_results=5,
		text={"max_characters": 1000},
		start_published_date=(datetime.now() + timedelta(days=-1)).replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z",
		end_published_date=datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).isoformat() + "Z",
	)
	return results.results


def get_old_and_new_info(news_info):
	system_prompt = """You are a helpful assistant. 
	You will be given a news story from which you need to tell me what the subject is, what the prompt is, 
	and what the existing ground truth was and what the new ground truth will be.
	Make the output follow the format in the JSON linked. """
	
	response = open_ai_client.chat.completions.create(
		model="gpt-4o-2024-08-06",
		messages=[
			{
			"role": "system",
			"content": [
				{
				"type": "text",
				"text": system_prompt
				}
			]
			},
			{
			"role": "user",
			"content": [ 
				{
				"type": "text",
				"text": news_info
				}
			]
			}
		],
		temperature=1,
		max_tokens=5000,
		top_p=1,
		frequency_penalty=0,
		presence_penalty=0,
		response_format={
			"type": "json_schema",
			"json_schema": {
			"name": "new_fact_dataset",
			"strict": False,
			"schema": {
				"subject": "The subject of the fact that has its ground truth changing",
				"prompt": "A one line summary of the fact that is changing",
				"ground_truth": "What the previous value for the fact used to be.",
				"target_new": "What the new value for the fact will be now",
				"type": "object",
				"properties": {},
				"required": []
			}
			}
		}
		)
	output = response.choices[0].message.content
	return json.loads(output)
	# return response

# Main function to build the CSV
def build_csv(filename, topics):
	df = pd.DataFrame(columns=["subject", "prompt", "ground_truth", "target_new"])
	
	for topic in topics:
		results = search_info(topic)
		for result in results:
			# new_search = search_and_validate("results")
			if result:
				new_info = result.text
				json_response = get_old_and_new_info(new_info)
				df = df._append({"subject": json_response["subject"], "prompt": json_response["prompt"], "ground_truth": json_response["ground_truth"], "target_new": json_response["target_new"]}, ignore_index=True)
	print(df)
	df.to_csv(filename, index=False)

# List of topics or facts to search for
topics = [
	"Latest news and advancements in Politics",
	"Latest news in sport"
	"Artificial intelligence advancements",
	"Climate change impact",
	"Developments in quantum computing",
	"Recent breakthroughs in renewable energy"
]

# Build the CSV
build_csv("llm_update_dataset.csv", topics)

print("CSV file 'llm_update_dataset.csv' has been created.")
