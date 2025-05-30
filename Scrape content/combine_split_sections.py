from pathlib import Path
import re

def get_section_number(filename: str) -> int:
    """Extract the section number from the filename."""
    match = re.match(r'(\d+)_', filename)
    if match:
        return int(match.group(1))
    return float('inf')  # Put files without numbers at the end

def combine_sections(input_dir: str, output_file: str):
    """Combine all split section files back into a single file."""
    # Get all txt files from the input directory
    input_path = Path(input_dir)
    section_files = list(input_path.glob('*.txt'))
    
    # Sort files by their section number
    section_files.sort(key=lambda x: get_section_number(x.name))
    
    # Combine all files
    with open(output_file, 'w', encoding='utf-8') as outfile:
        for section_file in section_files:
            # Read the content of each section file
            with open(section_file, 'r', encoding='utf-8') as infile:
                content = infile.read().strip()
                
                # Write the content to the output file
                outfile.write(content)
                
                # Add extra newlines between sections
                if section_file != section_files[-1]:  # Don't add newlines after the last section
                    outfile.write('\n\n\n')
    
    print(f"Combined {len(section_files)} sections into {output_file}")

if __name__ == "__main__":
    input_dir = "split_sections"  # Directory containing the split section files
    output_file = "combined_knowledge_base.txt"  # Output file name
    
    combine_sections(input_dir, output_file)
