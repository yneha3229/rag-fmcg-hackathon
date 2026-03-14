import wikipediaapi

wiki = wikipediaapi.Wikipedia(
    language='en',
    user_agent='rag_project/1.0'
)

# Pages to fetch
pages = [
    "Fast-moving consumer goods",
    "Supply chain management", 
    "Demand forecasting",
    "Inventory management",
    "Artificial intelligence",
    "Warehouse management system",
    "Logistics",
    "Retail"
]

# Fetch and save
with open("documents/fmcg.txt", "w", encoding="utf-8") as f:
    for page_name in pages:
        page = wiki.page(page_name)
        if page.exists():
            f.write(f"=== {page_name} ===\n")
            f.write(page.summary)  # just the summary, not full article
            f.write("\n\n")
            print(f"✅ Fetched: {page_name}")
        else:
            print(f"❌ Not found: {page_name}")

print("Done! Check documents/fmcg.txt")