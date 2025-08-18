import json
import os

# Path to your JSON
json_path = "src/content/books/readinglist.json"
output_dir = "src/content/books/"

# Load JSON
with open(json_path, "r", encoding="utf-8") as f:
    books = json.load(f)

# Make sure output directory exists
os.makedirs(output_dir, exist_ok=True)

for book in books:
    slug = book.get("slug")
    title = book.get("title")
    author = book.get("author")
    tags = book.get("tags", [])

    # Build MDX frontmatter
    frontmatter = [
        "---",
        f'slug: "{slug}"',
        f'title: "{title}"',
        f'author: "{author}"',
        # readYear is left blank for you to fill in
        f"readYear: ",
        f"tags: {tags}",
        "---",
        "",
        f"_Notes about **{title}** will go here._",
        ""
    ]

    mdx_content = "\n".join(frontmatter)

    # File path
    file_path = os.path.join(output_dir, f"{slug}.mdx")

    # Write file
    with open(file_path, "w", encoding="utf-8") as out:
        out.write(mdx_content)

    print(f"✅ Created {file_path}")
