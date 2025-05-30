import re

def clean_article_content(content):
    # Remove any "START" or "END" markers that appear in the content
    content = re.sub(r'str\("START"\)|str\("END"\)', '', content)
    content = re.sub(r'START|END', '', content)
    
    # Remove any duplicate URLs (keep only the first one)
    lines = content.split('\n')
    url_line = None
    cleaned_lines = []
    
    for line in lines:
        if line.startswith('URL: '):
            if url_line is None:
                url_line = line
                cleaned_lines.append(line)
        else:
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split by separator lines
    sections = re.split(r'={80}\n\n', content)
    
    # Process each section
    cleaned_sections = []
    for section in sections:
        if not section.strip():  # Skip empty sections
            continue
            
        # Clean the section content
        cleaned_content = clean_article_content(section)
        if cleaned_content.strip():  # Only add non-empty sections
            cleaned_sections.append(cleaned_content)
    
    # Write cleaned content to new file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n\n'.join(cleaned_sections))

if __name__ == '__main__':
    input_file = 'extracted_articles.txt'
    output_file = 'cleaned_extracted_articles.txt'
    process_file(input_file, output_file)
    print(f"Cleaned content has been written to {output_file}") 