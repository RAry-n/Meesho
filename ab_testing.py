import random
import statistics


class ABTestEngine:
def __init__(self, variants):
self.variants = variants # list of photo/description variants
self.data = {variant: [] for variant in variants}


def assign_variant(self, user_id):
# Random assignment with hash bucketing
idx = hash(user_id) % len(self.variants)
return self.variants[idx]


def record_interaction(self, variant, outcome):
self.data[variant].append(outcome)


def best_variant(self):
scores = {v: statistics.mean(self.data[v]) if self.data[v] else 0 for v in self.variants}
return max(scores.items(), key=lambda x: x[1])
