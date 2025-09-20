from ab_test import ABTestEngine
from search import SearchEngine
from visualization3d import Product3DRenderer
from chatbot import Chatbot
from gamification import GamificationEngine


class DummyDB:
def search_products(self, query):
return [{"id": 1, "tags": ["red", "saree"], "category": "saree", "brand": "Local"}]


def get_user_profile(self, user_id):
return {"preferred_categories": ["saree"], "preferred_brands": ["Local"]}


def search_with_filters(self, query, profile):
return [{"id": 2, "tags": ["saree", "traditional"], "category": "saree", "brand": "Local"}]


def run_demo():
db = DummyDB()


# AB Testing
ab = ABTestEngine(["photo1", "photo2"])
ab.record_interaction("photo1", 1)
ab.record_interaction("photo2", 0)
print("Best Variant:", ab.best_variant())


# Search
search_engine = SearchEngine(db)
results = search_engine.personalized_search("red saree", user_id=101)
print("Search Results:", results)


# 3D Visualization
renderer = Product3DRenderer()
print(renderer.generate_3d_model(["img1.jpg", "img2.jpg"]))


# Chatbot
bot = Chatbot(search_engine)
print("Chatbot Results:", bot.chatbot_search("show me red saree under ₹500", 101))


# Gamification
game = GamificationEngine()
print(game.complete_task(101, "Visit 3 shops"))
print("Leaderboard:", game.get_leaderboard())


if __name__ == "__main__":
run_demo()
