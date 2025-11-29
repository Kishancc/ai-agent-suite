# 🤖 AI Agent Suite – Sales, Marketing & Support

This project implements a **multi-agent AI system** with three agents in a single Gradio interface:

1. 🛟 **Support Assistant** – Answers FAQs and escalates complex queries  
2. 🛒 **Product Recommendation Agent** – Suggests relevant products from a large catalog  
3. 📣 **Social Media Content Generator** – Creates captions and a 7-day content plan  

It is built for the **Rooman AI Agent Development Challenge** under the **Sales, Marketing & Support** category.

---

## 🔍 Overview of the Agent Suite

### 1️⃣ Support Assistant

- Uses a FAQ knowledge base stored in `faqs.csv`
- Converts the user’s question into a TF-IDF vector
- Finds the most similar FAQ using **cosine similarity**
- Sends the FAQ context + question to **OpenAI GPT (gpt-4o-mini)** to generate a natural, friendly answer
- If similarity is below a threshold, the agent marks the query as **“escalated to human support”**

### 2️⃣ Product Recommendation Agent

- Uses a product catalog from `products.csv` (e.g., 5000 products)
- Filters products by:
  - Category
  - Maximum budget
  - Minimum rating
- Selects the top products based on rating and price
- Sends the user’s need + product details to GPT to generate a **simple explanation** of why these products are recommended

### 3️⃣ Social Media Content Generator

- Takes brand name, platform, campaign goal, tone, offer, and extra details
- Uses GPT to generate:
  - A hook
  - Main caption
  - Hashtags
  - Call-to-action
- Can also generate a **7-day content plan** with daily themes and formats

---

## ✨ Features

- Unified Gradio UI with **three dedicated tabs**
- **FAQ retrieval** using TF-IDF + cosine similarity (no expensive embeddings)
- Product filtering based on **realistic catalog fields** (category, price, rating)
- GPT-powered explanations and social content
- Uses **CSV datasets**, making it easy to plug in real company data

### ⚠ Limitations

- FAQ matching is keyword-based; extremely vague queries may produce weaker matches
- No persistent user login / session history
- Live demo via Colab + Gradio `share=True` only runs while the notebook is active
- No direct integration with external CRMs, ticketing tools, or e-commerce backends (yet)

---

## 🧠 Tech Stack & APIs Used

- **Language Model**: OpenAI `gpt-4o-mini` (Chat Completions API)
- **ML / Retrieval**: `scikit-learn` TF-IDF vectorizer + cosine similarity
- **UI**: Gradio
- **Data Handling**: Pandas, NumPy
- **Datasets**:
  - `faqs.csv` → columns: `id, category, question, answer`
  - `products.csv` → columns: `product_id, name, category, subcategory, brand, price, rating, tags, description`

---

## 📁 Project Structure

```text
ai-agent-suite/
├── app_gradio.py        # Main Gradio application
├── faqs.csv             # FAQ dataset
├── products.csv         # Product dataset
├── architecture.png     # Architecture diagram image
├── notebook.ipynb       # (Optional) Colab notebook used for development
├── requirements.txt     # Python dependencies
└── README.md            # This documentation
