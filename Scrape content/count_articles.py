import re

def count_articles(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            # Look for the pattern "> **URL:** <url>"
            pattern = r'>\s*\*\*URL:\*\*\s*<[^>]+>'
            matches = re.findall(pattern, content)
            return len(matches)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return 0
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return 0

if __name__ == "__main__":
    file_path = "Scrape content/knowledge base llms.txt"
    article_count = count_articles(file_path)
    print(f"Total number of articles found: {article_count}") 