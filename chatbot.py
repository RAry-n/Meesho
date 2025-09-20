import re
from search import SearchEngine


class Chatbot:
def __init__(self, search_engine):
self.search_engine = search_engine


def parse_query(self, query):
tags = {}
if "red" in query: tags["color"] = "red"
if "saree" in query: tags["category"] = "saree"
price_match = re.search(r'under ₹?(\d+)', query)
if price_match:
tags["price"] = {"$lt": int(price_match.group(1))}
return tags


def chatbot_search(self, query, user_id):
tags = self.parse_query(query)
return self.search_engine.personalized_search(" ".join(tags.values()), user_id)
