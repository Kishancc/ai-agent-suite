import os
import numpy as np
import pandas as pd
import gradio as gr

from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------------
# 0. OpenAI client
# -----------------------------
def get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set.\n\n"
            "Set it before running the app, for example:\n"
            "  import os\n"
            "  os.environ['OPENAI_API_KEY'] = 'sk-...'\n"
        )
    return OpenAI(api_key=api_key)


client = get_openai_client()


# -----------------------------
# 1. Load Datasets from CSV
# -----------------------------
def load_faq_csv(path: str = "faqs.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"FAQ file '{path}' not found. "
            f"Make sure faqs.csv is in the same directory as this script / notebook."
        )
    df = pd.read_csv(path)
    required_cols = {"id", "category", "question", "answer"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"faqs.csv must contain columns: {required_cols}. "
            f"Found: {df.columns.tolist()}"
        )
    return df


def load_products_csv(path: str = "products.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Products file '{path}' not found. "
            f"Make sure products.csv is in the same directory as this script / notebook."
        )
    df = pd.read_csv(path)
    required_cols = {
        "product_id",
        "name",
        "category",
        "subcategory",
        "brand",
        "price",
        "rating",
        "tags",
        "description",
    }
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"products.csv must contain columns: {required_cols}. "
            f"Found: {df.columns.tolist()}"
        )
    return df


faq_df = load_faq_csv("faqs.csv")
product_df = load_products_csv("products.csv")


# -----------------------------
# 2. Support Assistant – TF-IDF RAG
# -----------------------------
# Safely combine question + answer as string
faq_df["question"] = faq_df["question"].astype(str)
faq_df["answer"] = faq_df["answer"].astype(str)
faq_df["combined"] = faq_df["question"] + " " + faq_df["answer"]

vectorizer = TfidfVectorizer()
faq_tfidf = vectorizer.fit_transform(faq_df["combined"])


def support_assistant(query: str):
    """
    Uses TF-IDF + cosine similarity to find the closest FAQ and then
    asks the LLM to respond using that context.
    Returns: (answer_text, similarity_score_str)
    """
    query = (query or "").strip()
    if not query:
        return "Please enter a question.", "0.00"

    # --- Similarity search over FAQ ---
    try:
        q_vec = vectorizer.transform([query])
        sims = cosine_similarity(q_vec, faq_tfidf)[0]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])

        row = faq_df.iloc[best_idx]
        context = f"Q: {row['question']}\nA: {row['answer']}"
    except Exception as e:
        return f"Error while searching FAQ data: {e}", "0.00"

    escalation = best_score < 0.20  # threshold for escalation

    prompt = f"""
You are a helpful customer support assistant for an e-commerce store.

Customer question:
{query}

FAQ context:
{context}

If the answer is clearly present, respond using it.
If it's not clear, say you will escalate to a human agent.
"""

    # --- Call OpenAI Chat API ---
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        answer = resp.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}", f"{best_score:.2f}"

    if escalation:
        answer += (
            "\n\n📩 This looks complex. I've escalated it to a human support agent."
        )

    return answer, f"{best_score:.2f}"


def gr_support(query: str):
    return support_assistant(query)


# -----------------------------
# 3. Product Recommendation Agent
# -----------------------------
def recommend_products(category: str, budget: float | None, min_rating: float):
    """
    Filter products by category, budget, and rating.
    Returns top 3 by rating (and then price).
    """
    df = product_df.copy()

    # Normalize column types
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["price", "rating"])

    if category and category != "Any":
        df = df[df["category"] == category]

    if budget is not None and budget > 0:
        df = df[df["price"] <= budget]

    df = df[df["rating"] >= min_rating]

    if df.empty:
        df = product_df.copy()
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
        df = df.dropna(subset=["price", "rating"])
        df = df.nlargest(3, "rating")
    else:
        df = df.sort_values(["rating", "price"], ascending=[False, True]).head(3)

    return df


def explain_recommendations(user_need: str, rec_df: pd.DataFrame) -> str:
    """
    Use LLM to explain why the recommended products are suitable.
    """
    if rec_df.empty:
        return "No products found."

    products_text = ""
    for _, row in rec_df.iterrows():
        products_text += (
            f"- {row['name']} ({row['brand']}): ₹{row['price']}, "
            f"⭐ {row['rating']} – {row['description']}\n"
        )

    prompt = f"""
You are an AI product recommendation expert.

User need:
{user_need if user_need and user_need.strip() else "Not specified"}

Recommended products:
{products_text}

Explain in simple, friendly language why these products are suitable choices.
Keep it short and non-technical.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def gr_product(category, budget, min_rating, user_need):
    try:
        budget_val = float(budget) if budget is not None else None
    except (TypeError, ValueError):
        budget_val = None

    recs = recommend_products(category, budget_val, float(min_rating))
    explanation = explain_recommendations(user_need, recs)

    display_df = recs[
        ["name", "brand", "category", "price", "rating", "description"]
    ].reset_index(drop=True)

    return display_df, explanation


# -----------------------------
# 4. Social Media Content Agent
# -----------------------------
def generate_social_post(
    brand, platform, goal, tone, offer, extra
):
    prompt = f"""
You are a professional social media content creator.

Brand: {brand}
Platform: {platform}
Goal: {goal}
Tone: {tone}
Offer: {offer or "None"}
Extra details: {extra or "None"}

Generate:
1. A hook (1 line)
2. Main caption (3–5 short lines)
3. 8–12 relevant hashtags
4. A short call-to-action

Format it nicely.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def generate_content_plan(brand, platform, goal, days: int = 7):
    prompt = f"""
Create a {days}-day content plan for:

Brand: {brand}
Platform: {platform}
Goal: {goal}

For each day, provide:
- Theme/Idea
- Recommended format (post, reel, story, carousel, etc.)
- 1-line description

Return it as a bullet list, one bullet per day.
"""
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"Error calling OpenAI API: {e}"


def gr_social_single(brand, platform, goal, tone, offer, extra):
    return generate_social_post(brand, platform, goal, tone, offer, extra)


def gr_social_plan(brand, platform, goal):
    return generate_content_plan(brand, platform, goal, days=7)


# -----------------------------
# 5. Build Gradio UI
# -----------------------------
product_categories = (
    sorted(product_df["category"].dropna().astype(str).unique().tolist())
    if "category" in product_df.columns
    else []
)
categories = ["Any"] + product_categories

with gr.Blocks(theme="soft") as demo:
    gr.Markdown(
        "# 🤖 AI Agent Suite – Sales, Marketing & Support\n"
        "Three agents in one app: **Support Assistant**, **Product Recommender**, "
        "and **Social Media Content Generator**."
    )

    # --- Support Assistant Tab ---
    with gr.Tab("🛟 Support Assistant"):
        gr.Markdown("Ask any question related to orders, payments, returns, or account.")
        support_input = gr.Textbox(
            label="Customer Question",
            lines=3,
            placeholder="Example: How can I track my order?",
        )
        support_answer = gr.Textbox(label="AI Answer")
        support_conf = gr.Textbox(label="Similarity Score", interactive=False)

        support_button = gr.Button("Get Answer")
        support_button.click(
            fn=gr_support,
            inputs=support_input,
            outputs=[support_answer, support_conf],
        )

    # --- Product Recommendation Tab ---
    with gr.Tab("🛒 Product Recommendation"):
        gr.Markdown("Get product suggestions based on category, budget, and rating.")

        with gr.Row():
            prod_cat = gr.Dropdown(
                choices=categories, value="Any", label="Category"
            )
            prod_budget = gr.Number(value=5000, label="Max Budget (₹)")
            prod_rating = gr.Slider(
                minimum=1.0, maximum=5.0, value=4.0, step=0.1, label="Minimum Rating"
            )

        user_need = gr.Textbox(
            label="User Need (optional but recommended)",
            lines=3,
            placeholder="Example: I need a lightweight laptop for online classes.",
        )

        prod_table = gr.Dataframe(
            label="Top Recommendations",
            headers=[
                "name",
                "brand",
                "category",
                "price",
                "rating",
                "description",
            ],
            row_count=3,
            col_count=6,
        )
        prod_explanation = gr.Textbox(label="Why These Products?", lines=6)

        prod_button = gr.Button("Recommend Products")
        prod_button.click(
            fn=gr_product,
            inputs=[prod_cat, prod_budget, prod_rating, user_need],
            outputs=[prod_table, prod_explanation],
        )

    # --- Social Media Generator Tab ---
    with gr.Tab("📣 Social Media Generator"):
        gr.Markdown("Generate captions and a 7-day content plan.")

        with gr.Row():
            brand = gr.Textbox(label="Brand Name", value="ShopSmart")
            platform = gr.Dropdown(
                choices=["Instagram", "Facebook", "LinkedIn", "Twitter/X", "YouTube"],
                value="Instagram",
                label="Platform",
            )
        with gr.Row():
            tone = gr.Dropdown(
                choices=["Friendly", "Professional", "Bold", "Fun", "Educational"],
                value="Friendly",
                label="Tone",
            )
            goal = gr.Textbox(
                label="Campaign Goal",
                value="Increase awareness for our festive sale.",
            )

        offer = gr.Textbox(
            label="Offer / Highlight (optional)",
            value="Flat 30% OFF on selected items.",
        )
        extra = gr.Textbox(
            label="Extra Details (optional)",
            placeholder="Target audience, key products, etc.",
        )

        social_post_out = gr.Textbox(label="Generated Post", lines=10)
        social_plan_out = gr.Textbox(label="7-Day Content Plan", lines=12)

        with gr.Row():
            btn_post = gr.Button("Generate Single Post")
            btn_plan = gr.Button("Generate 7-Day Content Plan")

        btn_post.click(
            fn=gr_social_single,
            inputs=[brand, platform, goal, tone, offer, extra],
            outputs=social_post_out,
        )

        btn_plan.click(
            fn=gr_social_plan,
            inputs=[brand, platform, goal],
            outputs=social_plan_out,
        )


# For local run: python app_gradio.py
if __name__ == "__main__":
    demo.launch()
