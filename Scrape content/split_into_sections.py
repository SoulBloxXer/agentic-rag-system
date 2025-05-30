import re
from pathlib import Path
from typing import Dict, List, Tuple

# Define the section mappings with their starting URLs and numbers
SECTION_URLS = {
    "1 Case Studies": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/case-study-celtic-fish-game",
    "2 Troubleshooting Guides": "https://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/finished-shrink-wrap-problems-splitting-seals",
    "3 L Sealers": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/all-in-one-chamber-machine-5540",
    "4 Side Sealers": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-stainless-steel-automatic-side-sealing-machine-overview",
    "5 Sleeve Sealers": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-stainless-steel-semi-automatic-sleeve-sealer",
    "6 Stainless Steel Sealers": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-stainless-steel-automatic-side-sealing-machine-2",
    "7 Wide Sealers": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-large-l-sealer-and-tunnel-fl7555",
    "8 Heat Shrink Tunnels": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-sm-8040-shrink-tunnel",
    "9 Label & Sleeve Shrink Tunnels": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-labeling-shrink-tunnel",
    "10 Pallet Stretch Wrapping": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/a-semi-automatic-turntable-stretch-wrapper-saves-one-year-s-salary",
    "11 Hand Held Strapping": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/machinery-guide-handheld-strapping-machine",
    "12 Strapping & Ram Bunding Machines": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/knowledge-base-article-semi-automatic-strapping-machine",
    "13 Conveyors": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/non-powered-conveyor",
    "14 Buying Guides": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/how-automatic-vertical-stretch-wrapper-machines-save",
    "15 Box Sealing": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/exploring-belt-driven-box-sealer-machines-an-in-depth-guide",
    "16 Friction Feeders": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/understanding-linear-friction-feeder-machines-a-comprehensive-overview",
    "17 Inkjet & Labelling Systems": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/understanding-inkjet-coder-machines",
    "18 Special Applications & Robotics": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/understanding-self-propelled-robot-stretch-wrapping-machines",
    "19 Eco Friendly": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/we-show-you-that-polyolefin-is-recyclable",
    "20 Helpful Tips": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/how-to-work-out-shrink-film-sizes",
    "21 Shrink Wrap Films": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/difference-between-types-of-shrink-wrap",
    "22 Machine Servicing And Maintenance": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/risk-assessing",
    "23 Machine Demo's": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/can-i-trial-a-free-shrinkwrap-machine-demo",
    "24 Machine Operator Manuals": "http://crm.shrinkwrappingsupplies.co.uk/knowledge-base/article/pdf-manuals-machine-operating-instructions"
}

def extract_url_from_section(section: str) -> str:
    """Extract the URL from a section's content."""
    url_match = re.search(r'> \*\*URL:\*\* <(.*?)>', section)
    if url_match:
        return url_match.group(1)
    return ""

def get_section_for_url(url: str) -> str:
    """Determine which section a URL belongs to based on the starting URLs."""
    # Find the first section URL that matches the start of the given URL
    for category, start_url in SECTION_URLS.items():
        if url.startswith(start_url.split('/article/')[0]):
            return category
    return "Uncategorized"

def add_section_metadata(section: str, category: str) -> str:
    """Add section metadata after the URL line."""
    # Split the content into lines
    lines = section.split('\n')
    
    # Find the URL line and ensure proper spacing after horizontal line
    for i, line in enumerate(lines):
        if '---' in line:
            # Add a newline after the horizontal line if it doesn't exist
            if i + 1 < len(lines) and lines[i + 1].strip():
                lines.insert(i + 1, '')
        elif '> **URL:**' in line:
            # Insert the section metadata right after the URL line
            lines.insert(i + 1, f'> **Section:** {category}')
            break
    
    # Join the lines back together
    return '\n'.join(lines)

def split_into_sections(content: str) -> List[str]:
    """Split the content into sections based on the delimiter."""
    # Split by the delimiter and clean up each section
    sections = []
    for section in content.split('---'):
        section = section.strip()
        if section:
            # Ensure the section starts with a horizontal line
            if not section.startswith('---'):
                section = '---\n' + section
            sections.append(section)
    return sections

def get_clean_section_name(category: str) -> str:
    """Remove the number prefix from the category name."""
    # Remove the number and any leading/trailing spaces
    return ' '.join(category.split()[1:])

def process_file(input_file: str, output_dir: str):
    """Process the input file and create separate files for each category."""
    # Create output directory if it doesn't exist
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Read the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split into sections
    sections = split_into_sections(content)
    
    # Group sections by category
    categorized_sections: Dict[str, List[str]] = {}
    seen_urls = set()  # To track duplicates
    
    current_section = "Uncategorized"
    
    for section in sections:
        url = extract_url_from_section(section)
        if not url:
            continue
            
        # Check if this URL starts a new section
        for category, start_url in SECTION_URLS.items():
            if url == start_url:
                current_section = category
                break
        
        # Skip if we've seen this URL before
        if url in seen_urls:
            continue
            
        seen_urls.add(url)
        
        # Add section metadata to the section content
        section_with_metadata = add_section_metadata(section, current_section)
        
        if current_section not in categorized_sections:
            categorized_sections[current_section] = []
        categorized_sections[current_section].append(section_with_metadata)
    
    # Write each category to a separate file
    for category, sections in categorized_sections.items():
        # Create a safe filename from the category name (including the number)
        safe_filename = category.lower().replace(' & ', '_and_').replace(' ', '_')
        output_file = output_path / f"{safe_filename}.txt"
        
        # Get clean section name for the wrapper
        clean_section_name = get_clean_section_name(category)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # Add section wrapper at the start with clean name
            f.write(f"=== START of {clean_section_name} ===\n\n")
            
            # Write the content
            f.write('\n\n'.join(sections))
            
            # Add section wrapper at the end with clean name
            f.write(f"\n\n=== END of {clean_section_name} ===")
        
        print(f"Created {output_file} with {len(sections)} sections")

if __name__ == "__main__":
    input_file = "knowledge base llms.txt"  # Your input file
    output_dir = "split_sections"  # Directory where split files will be saved
    
    process_file(input_file, output_dir)
