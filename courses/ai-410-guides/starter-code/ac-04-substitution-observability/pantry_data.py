"""
Puget Pantry — a grocery delivery service's catalog, customers, and one
day of orders. You don't need to change this file.
"""

# sku -> product. `allergens` uses the customer allergy labels below.
CATALOG = {
    "milk-whole":    {"name": "Whole milk, 1/2 gal",     "category": "milk",   "price": 3.49, "allergens": {"dairy"},     "in_stock": False, "popularity": 9},
    "milk-oat":      {"name": "Oat milk, 1/2 gal",       "category": "milk",   "price": 4.29, "allergens": set(),         "in_stock": True,  "popularity": 8},
    "milk-almond":   {"name": "Almond milk, 1/2 gal",    "category": "milk",   "price": 3.99, "allergens": {"tree nuts"}, "in_stock": True,  "popularity": 9},
    "milk-soy":      {"name": "Soy milk, 1/2 gal",       "category": "milk",   "price": 3.79, "allergens": {"soy"},       "in_stock": True,  "popularity": 5},
    "bread-white":   {"name": "White sandwich bread",    "category": "bread",  "price": 2.99, "allergens": {"wheat"},     "in_stock": False, "popularity": 7},
    "bread-wheat":   {"name": "Whole wheat bread",       "category": "bread",  "price": 3.49, "allergens": {"wheat"},     "in_stock": True,  "popularity": 8},
    "bread-sourdough": {"name": "Artisan sourdough loaf", "category": "bread", "price": 6.99, "allergens": {"wheat"},     "in_stock": True,  "popularity": 9},
    "pb-creamy":     {"name": "Creamy peanut butter",    "category": "spread", "price": 3.99, "allergens": {"peanuts"},   "in_stock": False, "popularity": 9},
    "spread-almond": {"name": "Almond butter",           "category": "spread", "price": 8.99, "allergens": {"tree nuts"}, "in_stock": True,  "popularity": 8},
    "spread-sunflower": {"name": "Sunflower seed butter", "category": "spread", "price": 5.49, "allergens": set(),        "in_stock": True,  "popularity": 4},
    "yogurt-greek":  {"name": "Greek yogurt, plain",     "category": "yogurt", "price": 4.99, "allergens": {"dairy"},     "in_stock": False, "popularity": 8},
    "yogurt-coconut": {"name": "Coconut yogurt",         "category": "yogurt", "price": 5.29, "allergens": set(),         "in_stock": True,  "popularity": 6},
    "yogurt-regular": {"name": "Whole-milk yogurt",      "category": "yogurt", "price": 3.99, "allergens": {"dairy"},     "in_stock": True,  "popularity": 7},
    "eggs-dozen":    {"name": "Large eggs, dozen",       "category": "eggs",   "price": 4.49, "allergens": {"eggs"},      "in_stock": True,  "popularity": 9},
    "apples":        {"name": "Gala apples, 3 lb",       "category": "fruit",  "price": 4.99, "allergens": set(),         "in_stock": True,  "popularity": 7},
}

# customer_id -> customer. Allergies are what the customer told Puget Pantry.
CUSTOMERS = {
    "K-11": {"name": "Priya", "allergies": {"tree nuts"}},
    "K-12": {"name": "Marcus", "allergies": set()},
    "K-13": {"name": "Hana", "allergies": {"soy"}},
    "K-14": {"name": "Diego", "allergies": {"dairy"}},
    "K-15": {"name": "Ruth", "allergies": {"peanuts", "tree nuts"}},
    "K-16": {"name": "Sam", "allergies": set()},
}

# One day of orders, in the order they were packed.
ORDERS = [
    {"order_id": "O-501", "customer_id": "K-12", "skus": ["milk-whole", "eggs-dozen"]},
    {"order_id": "O-502", "customer_id": "K-11", "skus": ["milk-whole", "apples"]},
    {"order_id": "O-503", "customer_id": "K-16", "skus": ["eggs-dozen", "apples"]},
    {"order_id": "O-504", "customer_id": "K-13", "skus": ["bread-white", "yogurt-greek"]},
    {"order_id": "O-505", "customer_id": "K-15", "skus": ["pb-creamy", "bread-white"]},
    {"order_id": "O-506", "customer_id": "K-14", "skus": ["yogurt-greek", "apples"]},
    {"order_id": "O-507", "customer_id": "K-12", "skus": ["pb-creamy"]},
    {"order_id": "O-508", "customer_id": "K-16", "skus": ["bread-white", "milk-whole", "yogurt-greek"]},
    {"order_id": "O-509", "customer_id": "K-11", "skus": ["eggs-dozen"]},
    {"order_id": "O-510", "customer_id": "K-14", "skus": ["milk-whole", "bread-white"]},
]
