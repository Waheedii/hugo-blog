# openAi.py

import os
import json
import re
import time
import frontmatter
from datetime import datetime
from huggingface_hub import InferenceClient
import google.generativeai as genai

# --- NEW: Import our settings from the config file ---
import config

# --- HELPER: Load the master prompt from its dedicated file ---
def load_prompt_template(filepath: str) -> str:
    """Loads the prompt template from a text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"🔥 CRITICAL ERROR: Prompt template file not found at '{filepath}'")
        exit() # Exit the script if the prompt is missing

# --- INITIALIZATION & CONSTANTS ---
# Load the prompt using our new helper
PROMPT_TEMPLATE = load_prompt_template(config.PROMPT_TEMPLATE_FILE)

# Configure APIs using keys from the config file
genai.configure(api_key=config.GEMINI_API_KEY)
INFERENCE_CLIENT = InferenceClient(token=config.HF_API_TOKEN)

# Create directories using paths from the config file
os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.IMAGE_OUTPUT_DIR, exist_ok=True)


# --- IMAGE GENERATION FUNCTION ---
def generate_and_download_image(prompt: str, slug: str) -> str:
    """Generates an image using the huggingface_hub InferenceClient and saves it."""
    print(f"🎨 Sending prompt to Hugging Face: '{prompt}'")
    try:
        image = INFERENCE_CLIENT.text_to_image(
            prompt, 
            model=config.IMAGE_GENERATION_MODEL
        )
        local_filename = f"{slug}.jpg"
        local_filepath = os.path.join(config.IMAGE_OUTPUT_DIR, local_filename)
        image.save(local_filepath)
        hugo_image_path = f"/images/{local_filename}"
        print(f"💾 Image saved locally to: {local_filepath}")
        return hugo_image_path
    except Exception as e:
        print(f"🔥 Hugging Face Client Error: {e}")
        return None

# --- INTERNAL LINKING FUNCTIONS ---
def update_link_map(article_data: dict):
    """Adds the newly created article's info to the central link map."""
    link_map = []
    if os.path.exists(config.LINK_MAP_FILE):
        with open(config.LINK_MAP_FILE, "r", encoding="utf-8") as f:
            try:
                link_map = json.load(f)
            except json.JSONDecodeError:
                pass

    new_entry = {
        "slug": f"/{article_data['slug']}",
        "anchors": article_data.get("anchors", [])
    }
    link_map = [entry for entry in link_map if entry["slug"] != new_entry["slug"]]
    link_map.append(new_entry)
    with open(config.LINK_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(link_map, f, indent=2, ensure_ascii=False)

def apply_internal_links(filepath: str, current_slug: str):
    """Reads an article and injects internal links from the link map."""
    if not os.path.exists(config.LINK_MAP_FILE):
        return
    with open(config.LINK_MAP_FILE, "r", encoding="utf-8") as f:
        link_map = json.load(f)

    post = frontmatter.load(filepath)
    content = post.content
    links_added = 0
    max_links = 2

    for link_info in link_map:
        if link_info["slug"].strip('/') == current_slug:
            continue
        for anchor in link_info.get("anchors", []):
            if links_added >= max_links:
                break
            # Using a more robust regex to avoid linking inside existing links or markdown
            pattern = re.compile(r'(?<![\[(])\b' + re.escape(anchor) + r'\b(?![])])', re.IGNORECASE)
            new_content, count = pattern.subn(f'[{anchor}]({link_info["slug"]})', content, 1)
            if count > 0:
                content = new_content
                links_added += count
                break # Move to next link_info after one anchor is found
        if links_added >= max_links:
            break

    if links_added > 0:
        post.content = content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))
        print(f"🔗 {links_added} internal link(s) applied.")

# --- ARTICLE GENERATION & SAVING ---
def generate_article(article_data: dict, image_path: str) -> str:
    """Generates the full article content as a Markdown string."""
    try:
        current_date = datetime.now().strftime("%Y-%m-%d")
        categories_block = "\n".join([f"  - {c}" for c in article_data.get("categories", [])])
        tags_block = "\n".join([f"  - {t}" for t in article_data.get("tags", [])])

        prompt = PROMPT_TEMPLATE.format(
            title=article_data["title"],
            slug=article_data["slug"],
            description=article_data["description"],
            topic=article_data["topic"],
            date=current_date,
            image=image_path, # Using the dynamically generated image path
            categories_block=categories_block,
            tags_block=tags_block
        )
        model = genai.GenerativeModel(config.ARTICLE_GENERATION_MODEL)
        response = model.generate_content(prompt)
        time.sleep(2)
        return response.text
    except Exception as e:
        print(f"⚠️ Article generation failed: {e}")
        return None

def save_article(content: str, article_data: dict) -> str:
    """Cleans AI content, adds correct front matter, and saves the file."""
    try:
        # 1. Clean the raw AI output (remove ```markdown and extra front matter)
        content = re.sub(r"```markdown\s*", "", content)
        content = re.sub(r"```", "", content)
        cleaned_content = re.sub(r"^---.*?---\s*", "", content, flags=re.DOTALL).strip()

        # 2. Create a clean post object with the cleaned content
        post = frontmatter.loads(cleaned_content) # This just gets the content part
        
        # 3. Manually build the correct, clean front matter
        post.metadata = {
            "title": article_data["title"],
            "description": article_data["description"],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "categories": article_data.get("categories", []),
            "tags": article_data.get("tags", [])
        }
        
        filename = f"{article_data['slug']}.md"
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        
        # 4. Write the final file with clean metadata and content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))
        return filepath
    except Exception as e:
        print(f"🔥 Error saving article: {e}")
        return None


# --- MAIN WORKFLOW ---
def main():
    try:
        with open(config.TOPICS_FILE, "r", encoding="utf-8") as f:
            topics = json.load(f)
        if isinstance(topics, dict):
            topics = [topics]
    except Exception as e:
        print(f"⚠️ Could not read topics file: {e}")
        return

    if not topics:
        print("🎉 No topics left in the queue.")
        return

    article_data = topics[0]
    print("----------------------------------------------------")
    print(f"🚀 Starting daily article generation: {article_data['title']}")

    # Step 1: Generate hero image
    image_path = generate_and_download_image(article_data.get("image_prompt", article_data["topic"]), article_data["slug"])
    if not image_path:
        print("🔥 Image generation failed. Aborting run.")
        return

    # Step 2: Generate article content
    article_content = generate_article(article_data, image_path)
    if not article_content or len(article_content.strip()) < 200:
        print("⚠️ Article content generation failed or content is too short. Aborting.")
        return

    # Step 3: Save the final article file
    filepath = save_article(article_content, article_data)
    if not filepath:
        print("🔥 Saving the article failed. Aborting.")
        return

    # Step 4: Post-processing (linking)
    apply_internal_links(filepath, article_data["slug"])
    update_link_map(article_data)

    # Step 5: Update the topic queue
    remaining_topics = topics[1:]
    with open(config.TOPICS_FILE, "w", encoding="utf-8") as f:
        json.dump(remaining_topics, f, indent=2, ensure_ascii=False)

    print(f"✅ Article generated successfully: {filepath}")
    print(f"🔄 {len(remaining_topics)} topics left in queue.")
    print("----------------------------------------------------")

if __name__ == "__main__":
    main()