from collections import defaultdict


class SearchEngine:
def __init__(self, db):
self.db = db


def relevance_score(self, product, query):
return sum([1 for word in query.split() if word in product['tags']])


def personalization_score(self, product, user_profile):
score = 0
if product['category'] in user_profile['preferred_categories']:
score += 2
if product['brand'] in user_profile['preferred_brands']:
score += 1
return score


def personalized_search(self, query, user_id):
normal_results = self.db.search_products(query)
user_profile = self.db.get_user_profile(user_id)
personalized_results = self.db.search_with_filters(query, user_profile)


results = []
for product in {p['id']: p for p in normal_results + personalized_results}.values():
score = 0.6 * self.relevance_score(product, query) + \
0.4 * self.personalization_score(product, user_profile)
results.append((product, score))


return sorted(results, key=lambda x: x[1], reverse=True)
