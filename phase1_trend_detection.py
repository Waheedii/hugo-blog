import json
import random
from datetime import datetime

# CONFIG
DAILY_LIMIT = 1  # How many topics to pick each day

def load_curated_topics():
    with open("curated_topics.json", "r", encoding="utf-8") as f:
        data = json.load(f)
    return [topic for cat in data for topic in cat["topics"]]

def get_daily_topics(limit=DAILY_LIMIT):
    topics = load_curated_topics()
    return random.sample(topics, min(limit, len(topics)))

def save_daily_topics(topics):
    today = datetime.now().strftime("%Y-%m-%d")
    output = {
        "date": today,
        "topics": topics
    }
    with open("daily_topics.json", "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved {len(topics)} curated topics for {today}")

if __name__ == "__main__":
    topics = get_daily_topics()
    save_daily_topics(topics)
