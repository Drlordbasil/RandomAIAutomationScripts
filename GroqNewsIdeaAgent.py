import requests
import os
from openai import OpenAI

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_API_KEY)




class OpenAI_Advisor:

    def response(self):

        url = ('https://newsapi.org/v2/top-headlines?'
       'country=us&'
       'apiKey=817ce253061849c6abfdced81f961994')
        news_report = requests.get(url)
        news_report = news_report.json()
        print(news_report)
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {
                    "role": "system",
                    "content": f"""
                    You digest news articles and decide on the next big thing based on news articles.
                    You return with a project idea for python based on the news articles.
                    You can't make up any ideas that resemble any current technology or company.
                    it must be unique, it must profit, and it must be legal.

                     
                     """
                },
                {
                    "role": "user",
                    "content": f"{news_report}"
                }
            ]
        )
        print(response.choices[0].message.content)
        return response.choices[0].message.content
    def refine_idea(self, idea):
        response = client.chat.completions.create(
            model="llama3-70b-8192", # Or the appropriate model
            messages=[
                {"role": "system", "content": "You previously suggested this project idea. You are now asked to provide more details about potential target audiences and how it could generate revenue."},
                {"role": "user", "content": f"You previously suggested this project idea: {idea}. Can you provide more details about potential target audiences and how it could generate revenue?"}
            ]
        )
        return response.choices[0].message.content

# test time
advisor = OpenAI_Advisor()
idea = advisor.response()


