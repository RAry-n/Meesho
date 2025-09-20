class GamificationEngine:
def __init__(self):
self.user_points = {}
self.missions = [
{"task": "Visit 3 shops", "reward": 10},
{"task": "Buy 2 items", "reward": 20},
]


def complete_task(self, user_id, task):
mission = next((m for m in self.missions if m["task"] == task), None)
if mission:
self.user_points[user_id] = self.user_points.get(user_id, 0) + mission["reward"]
return f"User {user_id} completed '{task}' and earned {mission['reward']} points."
return "Invalid task"


def get_leaderboard(self):
return sorted(self.user_points.items(), key=lambda x: x[1], reverse=True)
