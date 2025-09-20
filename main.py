"""
This single-file reference implementation ties together components proposed in the
ideathon doc: A/B Testing for Sellers, Personalized Dual-Arm Search (left/right),
3D reconstruction pipeline (server-side stubs and export), NLQ chatbot + search
integration, and Gamification backend.

NOTE: This file is an architectural, runnable-like reference but uses placeholders
and abstractions for clarity. Replace stubs and adapters with production-grade
implementations (DB connections, model endpoints, cloud storage, inference infra,
etc.) before deployment.
"""

from __future__ import annotations
import hashlib
import json
import math
import random
import re
import threading
import time
import uuid
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Lightweight in-memory "DB" adapters for demonstration and performance notes
# ---------------------------------------------------------------------------

class InMemoryDB:
    """A simple thread-safe in-memory data store that mimics a subset of
    functionality of a relational DB + document DB for demonstration."""
    def __init__(self):
        self._lock = threading.RLock()
        self.products: Dict[str, Dict[str, Any]] = {}  # product_id -> product
        self.users: Dict[str, Dict[str, Any]] = {}
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        self.events: List[Dict[str, Any]] = []

    def insert_product(self, p: Dict[str, Any]):
        with self._lock:
            self.products[p['id']] = p

    def get_product(self, pid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.products.get(pid)

    def search_products_by_keyword(self, keyword: str) -> List[Dict[str, Any]]:
        """Naive full-text search for demonstration. In production, use
        Elasticsearch / OpenSearch / Postgres full-text indexes."""
        with self._lock:
            kw = keyword.lower()
            results = []
            for p in self.products.values():
                text = (p.get('title','') + ' ' + p.get('desc','')).lower()
                if kw in text:
                    results.append(p)
            return results

    def insert_user(self, u: Dict[str, Any]):
        with self._lock:
            self.users[u['id']] = u

    def get_user(self, uid: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.users.get(uid)

    def log_event(self, ev: Dict[str, Any]):
        with self._lock:
            self.events.append(ev)

    def create_ab_test(self, test_id: str, meta: Dict[str, Any]):
        with self._lock:
            self.ab_tests[test_id] = meta

    def get_ab_test(self, test_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self.ab_tests.get(test_id)


# global in-memory DB used by these examples
DB = InMemoryDB()

# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def now_ts():
    return datetime.utcnow().isoformat() + 'Z'


def uid(prefix: str='id') -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"

# simple cached sigmoid for score transform
@lru_cache(maxsize=4096)
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))

# ---------------------------------------------------------------------------
# 1) A/B Testing Engine for Sellers
# ---------------------------------------------------------------------------

@dataclass
class ABVariant:
    id: str
    payload: Dict[str, Any]
    impressions: int = 0
    clicks: int = 0
    purchases: int = 0

    def ctr(self) -> float:
        return (self.clicks / self.impressions) if self.impressions else 0.0

    def cr(self) -> float:
        return (self.purchases / self.impressions) if self.impressions else 0.0


@dataclass
class ABTest:
    id: str
    product_id: str
    field: str  # 'image' or 'description' etc.
    variants: Dict[str, ABVariant] = field(default_factory=dict)
    created_at: str = field(default_factory=now_ts)

    def add_variant(self, payload: Dict[str, Any]) -> str:
        vid = uid('v')
        self.variants[vid] = ABVariant(id=vid, payload=payload)
        return vid

    def sample_variant(self, user_id: Optional[str]=None) -> ABVariant:
        """Deterministic-ish bucketing: hash user_id + test_id to pick variant.
        This preserves consistent exposure for a user while distributing
        impressions."""
        if not self.variants:
            raise RuntimeError('No variants')
        variant_ids = sorted(self.variants.keys())
        if user_id is None:
            return self.variants[random.choice(variant_ids)]
        h = int(hashlib.sha256((user_id + self.id).encode()).hexdigest(), 16)
        chosen = variant_ids[h % len(variant_ids)]
        return self.variants[chosen]

    def record_impression(self, vid: str):
        self.variants[vid].impressions += 1

    def record_click(self, vid: str):
        self.variants[vid].clicks += 1

    def record_purchase(self, vid: str):
        self.variants[vid].purchases += 1

    def evaluate(self) -> Dict[str, Any]:
        """Return ranking and statistical metrics. This uses a simple Bayesian
        estimate (Laplace smoothing) for conversion estimates + uplift.
        In production, replace with a proper sequential testing library."""
        results = []
        for v in self.variants.values():
            alpha = 1 + v.purchases
            beta = 1 + (v.impressions - v.purchases)
            mean = alpha / (alpha + beta)
            results.append({
                'variant_id': v.id,
                'impressions': v.impressions,
                'clicks': v.clicks,
                'purchases': v.purchases,
                'estimated_cr': mean,
                'ctr': v.ctr(),
                'cr': v.cr()
            })
        # sort by estimated_cr desc
        results.sort(key=lambda x: x['estimated_cr'], reverse=True)
        return {
            'test_id': self.id,
            'product_id': self.product_id,
            'field': self.field,
            'variants': results,
            'evaluated_at': now_ts()
        }


class ABEngine:
    """High-level façade for creating and managing A/B tests."""
    def __init__(self, db: InMemoryDB):
        self.db = db

    def create_test(self, product_id: str, field: str, payloads: List[Dict[str, Any]]) -> str:
        test_id = uid('ab')
        test = ABTest(id=test_id, product_id=product_id, field=field)
        for p in payloads:
            test.add_variant(p)
        self.db.create_ab_test(test_id, {'meta': test})
        return test_id

    def get_test(self, test_id: str) -> Optional[ABTest]:
        data = self.db.get_ab_test(test_id)
        if data:
            return data['meta']
        return None


# ---------------------------------------------------------------------------
# 2) Personalized Dual-Arm Search (Left + Right)
# ---------------------------------------------------------------------------

class QueryOptimizer:
    """Component to optimize DB queries. For demonstration we craft a
    pseudo-query plan: try to use filters that reduce cardinality early and
    leverage indexes (simulated)."""

    @staticmethod
    def optimize_filters(filters: Dict[str, Any]) -> Dict[str, Any]:
        # Heuristic: move equality filters first, range filters next, text last
        eq = {k: v for k, v in filters.items() if not isinstance(v, dict)}
        rng = {k: v for k, v in filters.items() if isinstance(v, dict)}
        ordered = {**eq, **rng}
        return ordered


class PersonalizationModel:
    """A tiny scoring model that computes a personalization score given user
    profile and product metadata. Replace with logistic regression or
    LightGBM/Prod models in real systems."""
    def score(self, user: Dict[str, Any], product: Dict[str, Any]) -> float:
        # features: category match, price proximity, brand affinity
        score = 0.0
        if 'preferred_categories' in user and product.get('category') in user['preferred_categories']:
            score += 1.0
        # price affinity: prefer +/- 30% of user's avg_spend
        avg_spend = user.get('avg_spend', 0)
        price = product.get('price', 0)
        if avg_spend > 0:
            ratio = price / avg_spend
            score += max(0.0, 1.0 - abs(ratio - 1.0))
        # brand affinity
        if 'preferred_brands' in user and product.get('brand') in user['preferred_brands']:
            score += 0.5
        return sigmoid(score)


class DualArmSearch:
    """Brings left-arm (keyword) and right-arm (personalization) results and
    merges them with weighting logic. Demonstrates DB query optimization
    strategies via QueryOptimizer and caching."""
    def __init__(self, db: InMemoryDB, model: PersonalizationModel):
        self.db = db
        self.model = model
        self.cache = {}  # naive LRU could be added

    def left_arm_search(self, query: str, limit: int=50) -> List[Dict[str, Any]]:
        # optimized naive search
        products = self.db.search_products_by_keyword(query)
        # naive relevance by keyword count
        scored = []
        ql = query.lower().split()
        for p in products:
            text = (p.get('title','') + ' ' + p.get('desc','')).lower()
            score = sum(text.count(w) for w in ql)
            scored.append((p, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [p for p, s in scored[:limit]]

    def right_arm_search(self, query: str, user_id: str, limit: int=50) -> List[Tuple[Dict[str, Any], float]]:
        user = self.db.get_user(user_id) or {}
        # Example: start with left arm but re-score with personalization
        base = self.left_arm_search(query, limit=200)
        scored = []
        for p in base:
            score = self.model.score(user, p)
            scored.append((p, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def merge_results(self, left: List[Dict[str, Any]], right: List[Tuple[Dict[str, Any], float]], user_id: str) -> List[Dict[str, Any]]:
        # We will compute a blended score: alpha * left_relevance + beta * personalization
        alpha = 0.6
        beta = 0.4
        left_scores = {p['id']: (i, 1.0/(1+i)) for i,p in enumerate(left)}  # higher rank -> higher base
        right_scores = {p['id']: score for p, score in right}
        final = {}
        for p in left:
            pid = p['id']
            lscore = left_scores.get(pid, (999, 0))[1]
            pscore = right_scores.get(pid, 0)
            blended = alpha*lscore + beta*pscore
            final[pid] = (p, blended)
        for p, rscore in right:
            pid = p['id']
            if pid not in final:
                final[pid] = (p, beta*rscore)
        items = list(final.values())
        items.sort(key=lambda x: x[1], reverse=True)
        return [p for p, s in items]

    def search(self, query: str, user_id: Optional[str]=None, limit: int=20) -> List[Dict[str, Any]]:
        left = self.left_arm_search(query, limit=limit)
        if user_id:
            right = self.right_arm_search(query, user_id, limit=limit)
        else:
            right = []
        merged = self.merge_results(left, right, user_id or '')
        return merged[:limit]


# ---------------------------------------------------------------------------
# 3) 3D Reconstruction Pipeline (Server-side stubs + export helper)
# ---------------------------------------------------------------------------

class ThreeDReconstructor:
    """Server-side 3D reconstruction pipeline. This module is a stub that
    defines the expected API and export format. In a production flow, you
    would run photogrammetry / neural-rendering models (eg. COLMAP,
    NeRF/RegNeRF, or multi-view stereo) on a GPU cluster.

    This class provides:
     - validate_images(images)
     - schedule_reconstruction(product_id, images)
     - poll_status(task_id)
     - retrieve_glb(task_id)

    For demo purposes, the pipeline will synthesize a low-poly cube mesh and
    export a minimal GLB/ThreeJS-compatible JSON structure.
    """
    def __init__(self):
        self.tasks: Dict[str, Dict[str, Any]] = {}

    def validate_images(self, images: List[bytes]) -> bool:
        # Real checks: EXIF orientation, resolution thresholds, diversity of angles
        return len(images) >= 3

    def schedule_reconstruction(self, product_id: str, images: List[bytes]) -> str:
        if not self.validate_images(images):
            raise ValueError('Need at least 3 images from different angles')
        task_id = uid('3d')
        # simulate background processing
        self.tasks[task_id] = {
            'product_id': product_id,
            'status': 'queued',
            'created_at': now_ts(),
            'result': None
        }
        # spawn thread to simulate work
        threading.Thread(target=self._run_task, args=(task_id, images), daemon=True).start()
        return task_id

    def _run_task(self, task_id: str, images: List[bytes]):
        self.tasks[task_id]['status'] = 'processing'
        time.sleep(1 + random.random()*2)  # simulate compute
        # synthesize a simple GLB-like JSON structure
        glb = self._synthesize_glb(product_id=self.tasks[task_id]['product_id'])
        self.tasks[task_id]['status'] = 'done'
        self.tasks[task_id]['result'] = glb

    def _synthesize_glb(self, product_id: str) -> Dict[str, Any]:
        # Minimal cube mesh
        mesh = {
            'nodes': [{'name': f'{product_id}_root'}],
            'meshes': [{'name': 'cube'}],
            'buffers': {},
            'materials': [{'pbrMetallicRoughness': {'baseColorFactor': [1,1,1,1]}}]
        }
        return mesh

    def poll_status(self, task_id: str) -> Dict[str, Any]:
        return self.tasks.get(task_id, {'status': 'missing'})

    def retrieve_glb(self, task_id: str) -> Optional[Dict[str, Any]]:
        t = self.tasks.get(task_id)
        if t and t['status'] == 'done':
            return t['result']
        return None


# ---------------------------------------------------------------------------
# 4) NLQ Chatbot + Integration with Personalized Search
# ---------------------------------------------------------------------------

class NLQParser:
    """A modest rule-based NLQ parser for demonstration. In production, use
    transformer-based NER + dependency parse + entity linking (eg. spaCy,
    HuggingFace models) to extract intents and slots robustly across languages.
    """
    PRICE_PATTERN = re.compile(r'(under|below)\s*₹?(\d+(?:,\d{3})*)', re.I)
    RANGE_PATTERN = re.compile(r'(between)\s*₹?(\d+(?:,\d{3})*)\s*(and|to)\s*₹?(\d+(?:,\d{3})*)', re.I)

    def parse(self, text: str) -> Dict[str, Any]:
        text = text.lower()
        tags: Dict[str, Any] = {}
        # color
        for color in ['red', 'blue', 'green', 'black', 'white', 'yellow', 'pink']:
            if color in text:
                tags['color'] = color
                break
        # category (very simplified)
        for cat in ['saree', 'kurta', 'dress', 'shirt', 'jeans', 'mobile', 'earbuds', 'sofa', 'table']:
            if cat in text:
                tags['category'] = cat
                break
        # price
        m = self.PRICE_PATTERN.search(text)
        if m:
            tags['price'] = {'$lt': int(m.group(2).replace(',',''))}
        r = self.RANGE_PATTERN.search(text)
        if r:
            tags['price'] = {'$gte': int(r.group(2).replace(',','')), '$lte': int(r.group(4).replace(',',''))}
        # boolean modifiers
        if 'cheap' in text or 'affordable' in text:
            tags.setdefault('sort', 'price_asc')
        if 'best' in text or 'top' in text:
            tags.setdefault('sort', 'relevance')
        return tags


class Chatbot:
    """Integrates NLQ parsing and search. Exposes chat_search(query, user_id)"""
    def __init__(self, parser: NLQParser, searcher: DualArmSearch):
        self.parser = parser
        self.searcher = searcher

    def chat_search(self, user_id: str, query: str, limit: int=20) -> List[Dict[str, Any]]:
        tags = self.parser.parse(query)
        # build a query string for left arm if category present use it, else use raw text
        left_query = tags.get('category', query)
        # perform dual-arm search
        candidates = self.searcher.search(left_query, user_id=user_id, limit=200)
        # apply simple filters from tags
        filtered = []
        for p in candidates:
            ok = True
            price = p.get('price', 0)
            pr = tags.get('price')
            if pr:
                if '$lt' in pr and not (price < pr['$lt']): ok = False
                if '$gte' in pr and not (price >= pr['$gte']): ok = False
                if '$lte' in pr and not (price <= pr['$lte']): ok = False
            if tags.get('color') and tags['color'] not in p.get('colors', []):
                ok = False
            if ok:
                filtered.append(p)
            if len(filtered) >= limit:
                break
        return filtered


# ---------------------------------------------------------------------------
# 5) Gamification Backend
# ---------------------------------------------------------------------------

@dataclass
class PlayerState:
    user_id: str
    coins: int = 0
    badges: List[str] = field(default_factory=list)
    visited_nodes: List[str] = field(default_factory=list)
    last_active: str = field(default_factory=now_ts)


class GamificationEngine:
    """Stateful gamification engine that tracks player progress and issues
    rewards. Simple event-driven architecture with missions and a path graph."""
    def __init__(self):
        self.players: Dict[str, PlayerState] = {}
        # define a simple path graph: nodes are themed streets
        self.graph = [
            {'id': 'home', 'name': 'Home Street', 'reward': 5},
            {'id': 'saree', 'name': 'Saree Lane', 'reward': 10},
            {'id': 'electronics', 'name': 'Electro Bazaar', 'reward': 8},
            {'id': 'home_decor', 'name': 'Home Decor Plaza', 'reward': 6},
            {'id': 'checkout', 'name': 'Checkout Square', 'reward': 15}
        ]
        self.missions = {
            'visit_3': {'desc': 'Visit 3 shops', 'coins': 10},
            'first_purchase': {'desc': 'Complete first purchase', 'coins': 50}
        }

    def get_player(self, user_id: str) -> PlayerState:
        if user_id not in self.players:
            self.players[user_id] = PlayerState(user_id=user_id)
        return self.players[user_id]

    def visit_node(self, user_id: str, node_id: str) -> PlayerState:
        p = self.get_player(user_id)
        if node_id not in p.visited_nodes:
            p.visited_nodes.append(node_id)
            # reward for visiting new node
            node = next((n for n in self.graph if n['id'] == node_id), None)
            if node:
                p.coins += node['reward']
                DB.log_event({'type': 'gamify_visit', 'user': user_id, 'node': node_id, 'reward': node['reward'], 'ts': now_ts()})
        p.last_active = now_ts()
        # mission check: visit_3
        if len(p.visited_nodes) >= 3 and 'visit_3' not in p.badges:
            p.badges.append('visit_3')
            p.coins += self.missions['visit_3']['coins']
            DB.log_event({'type': 'mission_complete', 'user': user_id, 'mission': 'visit_3', 'ts': now_ts()})
        return p

    def complete_purchase(self, user_id: str, amount: float) -> PlayerState:
        p = self.get_player(user_id)
        # reward purchase
        p.coins += int(amount // 100)  # 1 coin per 100 spent
        if 'first_purchase' not in p.badges:
            p.badges.append('first_purchase')
            p.coins += self.missions['first_purchase']['coins']
            DB.log_event({'type': 'mission_complete', 'user': user_id, 'mission': 'first_purchase', 'ts': now_ts()})
        DB.log_event({'type': 'purchase_reward', 'user': user_id, 'amount': amount, 'coins': p.coins, 'ts': now_ts()})
        return p


# ---------------------------------------------------------------------------
# Example Data Population & Demonstration Runner (puts it all together)
# ---------------------------------------------------------------------------

def populate_demo_data(db: InMemoryDB):
    # Create demo products
    products = [
        {'id': 'p1', 'title': 'Red Cotton Saree', 'desc': 'Bright red saree', 'price': 499, 'category': 'saree', 'brand': 'LocalWeave', 'colors': ['red','pink']},
        {'id': 'p2', 'title': 'Blue Denim Jeans', 'desc': 'Slim fit jeans', 'price': 899, 'category': 'jeans', 'brand': 'DenimCo', 'colors': ['blue']},
        {'id': 'p3', 'title': 'Wireless Earbuds', 'desc': 'Noise cancelling earbuds', 'price': 1299, 'category': 'earbuds', 'brand': 'SoundX', 'colors': ['black']},
        {'id': 'p4', 'title': 'Green Kurta', 'desc': 'Festive kurta', 'price': 799, 'category': 'kurta', 'brand': 'Tradition', 'colors': ['green','black']},
    ]
    for p in products:
        db.insert_product(p)

    # Users
    users = [
        {'id': 'u1', 'name': 'Asha', 'preferred_categories': ['saree','kurta'], 'avg_spend': 600, 'preferred_brands': ['LocalWeave']},
        {'id': 'u2', 'name': 'Rohit', 'preferred_categories': ['earbuds','jeans'], 'avg_spend': 1000, 'preferred_brands': ['SoundX','DenimCo']},
    ]
    for u in users:
        db.insert_user(u)


def demo_flow():
    populate_demo_data(DB)
    # Setup components
    ab_engine = ABEngine(DB)
    model = PersonalizationModel()
    searcher = DualArmSearch(DB, model)
    nlq = NLQParser()
    bot = Chatbot(nlq, searcher)
    recon = ThreeDReconstructor()
    gamify = GamificationEngine()

    # 1) A/B test for product images
    test_id = ab_engine.create_test('p1', 'image', [
        {'url': 'https://cdn.example/p1_img1.jpg', 'meta': {'angle': 'front'}},
        {'url': 'https://cdn.example/p1_img2.jpg', 'meta': {'angle': 'side'}},
        {'url': 'https://cdn.example/p1_img3.jpg', 'meta': {'angle': 'drape'}},
    ])
    test = ab_engine.get_test(test_id)
    # simulate impressions
    for uid in ['u1','u2','anon1','anon2','u1','anon3']:
        v = test.sample_variant(uid)
        test.record_impression(v.id)
        if random.random() < 0.2:
            test.record_click(v.id)
        if random.random() < 0.05:
            test.record_purchase(v.id)
    print('A/B test evaluation:', json.dumps(test.evaluate(), indent=2))

    # 2) Personalized search
    print('\nSearch results for Asha (query: saree):')
    res = searcher.search('saree', user_id='u1')
    for r in res:
        print('-', r['id'], r['title'])

    # 3) 3D recon
    images = [b'img1', b'img2', b'img3']
    task = recon.schedule_reconstruction('p1', images)
    time.sleep(2)
    print('\n3D task status & result:', recon.poll_status(task))

    # 4) Chatbot NLQ
    print('\nChatbot query: "red sarees under ₹500"')
    bot_results = bot.chat_search('u1', 'red sarees under ₹500')
    for p in bot_results:
        print('-', p['id'], p['title'], p.get('price'))

    # 5) Gamification
    print('\nGamification demo:')
    gm = gamify.visit_node('u1', 'home')
    gm = gamify.visit_node('u1', 'saree')
    gm = gamify.visit_node('u1', 'electronics')
    gm = gamify.complete_purchase('u1', 560)
    print('Player state:', gm)


if __name__ == '__main__':
    demo_flow()
