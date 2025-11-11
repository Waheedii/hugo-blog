# openAi.py - A Multi-Agent Content Generation Pipeline (Final Corrected Version)

import os
import json
import re
import time
import frontmatter
from datetime import datetime
from huggingface_hub import InferenceClient
import google.generativeai as genai

# --- Import our settings from the config file ---
import config

# --- HELPER: Load prompt templates from their dedicated files ---
def load_prompt_template(filepath: str) -> str:
    """Loads a prompt template from a text file."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        print(f"🔥 CRITICAL ERROR: Prompt template file not found at '{filepath}'")
        exit()

# --- INITIALIZATION & CONSTANTS ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
WRITER_PROMPT_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, config.PROMPT_TEMPLATE_FILE)
EDITOR_PROMPT_TEMPLATE_PATH = os.path.join(SCRIPT_DIR, config.EDITOR_PROMPT_TEMPLATE_FILE)
TOPICS_FILE_PATH = os.path.join(SCRIPT_DIR, config.TOPICS_FILE)
LINK_MAP_FILE_PATH = os.path.join(SCRIPT_DIR, config.LINK_MAP_FILE)

WRITER_PROMPT_TEMPLATE = load_prompt_template(WRITER_PROMPT_TEMPLATE_PATH)
EDITOR_PROMPT_TEMPLATE = load_prompt_template(EDITOR_PROMPT_TEMPLATE_PATH)

genai.configure(api_key=config.GEMINI_API_KEY)
HF_CLIENT = InferenceClient(token=config.HF_API_TOKEN)

os.makedirs(config.OUTPUT_DIR, exist_ok=True)
os.makedirs(config.IMAGE_OUTPUT_DIR, exist_ok=True)
os.makedirs(config.DRAFTS_DIR, exist_ok=True)


# --- AGENT 1: The Writer (Gemini) ---
def generate_article_draft(article_data: dict) -> str:
    """Generates the initial raw article draft from Gemini."""
    print("✍️  Generating initial draft from Writer (Gemini)...")
    try:
        prompt = WRITER_PROMPT_TEMPLATE.format(topic=article_data["topic"])
        model = genai.GenerativeModel(config.ARTICLE_GENERATION_MODEL)
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        print(f"⚠️ Gemini draft generation failed: {e}")
        return None

# --- AGENT 2: The Editor (Llama 3.1) ---
def refine_article_with_llama(draft_content: str, article_data: dict) -> str:
    """Sends the draft to Llama 3.1 for refinement and image placeholder insertion."""
    print("🧐 Sending draft to Editor (Llama 3.1) for refinement...")
    try:
        system_prompt = EDITOR_PROMPT_TEMPLATE.format(topic=article_data["topic"])
        completion = HF_CLIENT.chat.completions.create(
            model=config.EDITOR_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": draft_content}
            ],
            max_tokens=4096
        )
        refined_text = completion.choices[0].message.content
        return refined_text
    except Exception as e:
        print(f"⚠️ Llama refinement failed: {e}")
        return None

# --- AGENT 3: The Producer (Python Script Functions) ---
def generate_and_download_image(prompt: str, image_filename: str) -> str:
    """Generates an image using Stable Diffusion and saves it."""
    print(f"🎨 Generating image for: '{prompt[:60]}...'")
    try:
        image = HF_CLIENT.text_to_image(prompt, model=config.IMAGE_GENERATION_MODEL)
        local_filepath = os.path.join(config.IMAGE_OUTPUT_DIR, image_filename)
        image.save(local_filepath)
        hugo_image_path = f"/images/{image_filename}"
        print(f"💾 Image saved: {hugo_image_path}")
        return hugo_image_path
    except Exception as e:
        print(f"🔥 Hugging Face image generation failed: {e}")
        return None

def process_article_images(refined_content: str, slug: str) -> str:
    """Finds image placeholders, generates images, and replaces them with Hugo shortcodes."""
    print("🖼️  Processing dynamic image placeholders...")
    placeholder_pattern = re.compile(r'\[IMAGE\|([\w-]+)\|([^\]]+)\]')
    placeholders = placeholder_pattern.findall(refined_content)

    if not placeholders:
        print("➡️ No dynamic image placeholders found in the refined text.")
        return refined_content

    processed_content = refined_content
    for image_id, prompt in placeholders:
        print(f"  -> Found placeholder ID: {image_id}")
        image_filename = f"{slug}-{image_id}.jpg"
        hugo_path = generate_and_download_image(prompt.strip(), image_filename)
        if hugo_path:
            caption = prompt.split(',')[0]
            shortcode = f'\n{{{{< figure src="{hugo_path}" caption="{caption}" >}}}}\n'
            placeholder_full_text = f'[IMAGE|{image_id}|{prompt}]'
            processed_content = processed_content.replace(placeholder_full_text, shortcode)
    return processed_content

def save_article(content_body: str, hero_image_path: str, article_data: dict) -> str:
    """Constructs the final markdown file with front matter and saves it."""
    try:
        post = frontmatter.Post(content_body)
        post.metadata = {
            "title": article_data["title"],
            "description": article_data["description"],
            "date": datetime.now().strftime("%Y-%m-%d"),
            "categories": article_data.get("categories", []),
            "tags": article_data.get("tags", []),
            "hero": hero_image_path
        }
        filename = f"{article_data['slug']}.md"
        filepath = os.path.join(config.OUTPUT_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))
        return filepath
    except Exception as e:
        print(f"🔥 Error saving the final article: {e}")
        return None

# --- SEO & Linking Functions ---
def update_link_map(article_data: dict):
    """Adds the newly created article's info to the central link map."""
    link_map = []
    if os.path.exists(LINK_MAP_FILE_PATH):
        with open(LINK_MAP_FILE_PATH, "r", encoding="utf-8") as f:
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
    with open(LINK_MAP_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(link_map, f, indent=2, ensure_ascii=False)

def apply_internal_links(filepath: str, current_slug: str):
    """Reads an article and injects internal links from the link map."""
    if not os.path.exists(LINK_MAP_FILE_PATH):
        return
    with open(LINK_MAP_FILE_PATH, "r", encoding="utf-8") as f:
        link_map = json.load(f)
    post = frontmatter.load(filepath)
    content = post.content
    links_added = 0
    max_links = 2
    for link_info in link_map:
        if link_info["slug"].strip('/') == current_slug:
            continue
        for anchor in link_info.get("anchors", []):
            if links_added >= max_links: break
            pattern = re.compile(r'(?<![\[(])\b' + re.escape(anchor) + r'\b(?![])])', re.IGNORECASE)
            new_content, count = pattern.subn(f'[{anchor}]({link_info["slug"]})', content, 1)
            if count > 0:
                content = new_content
                links_added += count
                break
        if links_added >= max_links: break
    if links_added > 0:
        post.content = content
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(frontmatter.dumps(post))
        print(f"🔗 {links_added} internal link(s) applied.")

# --- MAIN WORKFLOW: The Orchestrator (Single, Corrected Version) ---
def main():
    print("--- SCRIPT STARTED: Running main() function ---") # CHECKPOINT 1
    try:
        with open(TOPICS_FILE_PATH, "r", encoding="utf-8") as f:
            topics = json.load(f)
        if isinstance(topics, dict):
            topics = [topics]
        print(f"--- CHECKPOINT 2: Successfully loaded topics file. Found {len(topics)} topics. ---")
    except Exception as e:
        print(f"⚠️ Could not read topics file: {e}")
        return

    if not topics:
        print("--- CHECKPOINT 3: Topic list is empty. Exiting. ---")
        print("🎉 No topics left in the queue.")
        return

    article_data = topics[0]
    print("----------------------------------------------------")
    print(f"🚀 Starting multi-agent generation for: {article_data['title']}")

    draft_content = generate_article_draft(article_data)
    if not draft_content:
        print("🔥 Initial draft generation failed. Aborting."); return

    try:
        draft_filename = f"{datetime.now().strftime('%Y-%m-%d')}-{article_data['slug']}.md"
        draft_filepath = os.path.join(config.DRAFTS_DIR, draft_filename)
        with open(draft_filepath, "w", encoding="utf-8") as f:
            f.write(draft_content)
        print(f"📝 Draft saved for review: {draft_filepath}")
    except Exception as e:
        print(f"⚠️ Could not save draft file: {e}")

    refined_content_with_placeholders = refine_article_with_llama(draft_content, article_data)
    if not refined_content_with_placeholders:
        print("🔥 Llama refinement failed. Aborting."); return

    hero_prompt = article_data.get("image_prompt", f"A professional hero image for a blog post about {article_data['topic']}, vector art, minimalist style")
    hero_filename = f"{article_data['slug']}-hero.jpg"
    hero_image_path = generate_and_download_image(hero_prompt, hero_filename)
    if not hero_image_path:
        print("🔥 Hero image generation failed. Aborting."); return

    final_article_body = process_article_images(refined_content_with_placeholders, article_data['slug'])
    hero_shortcode = f'{{{{< figure src="{hero_image_path}" caption="Overview of {article_data["topic"]}" >}}}}\n\n'
    final_content_with_hero = hero_shortcode + final_article_body

    filepath = save_article(final_content_with_hero, hero_image_path, article_data)
    if not filepath:
        print("🔥 Saving the final article failed. Aborting."); return
    
    apply_internal_links(filepath, article_data['slug'])
    update_link_map(article_data)

    remaining_topics = topics[1:]
    with open(TOPICS_FILE_PATH, "w", encoding="utf-8") as f:
        json.dump(remaining_topics, f, indent=2, ensure_ascii=False)

    print(f"✅✅✅ Multi-agent article generated successfully: {filepath}")
    print(f"🔄 {len(remaining_topics)} topics left in queue.")
    print("----------------------------------------------------")

# This is the line that actually runs the main() function.
if __name__ == "__main__":
    main()