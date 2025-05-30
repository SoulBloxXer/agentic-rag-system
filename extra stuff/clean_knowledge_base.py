def clean_knowledge_base(input_file: str, output_file: str) -> None:
    """
    Clean the knowledge base file by removing category wrappers while preserving article content.
    The function processes the input file to:
    1. Remove all "=== START of ... ===" and "=== END of ... ===" category wrapper lines
    2. Preserve article content blocks separated by "---"
    3. Clean up excessive empty lines
    4. Save the cleaned content to a new file
    
    Args:
        input_file (str): Path to the input knowledge base file containing category wrappers
        output_file (str): Path where the cleaned knowledge base will be saved
    """
    # Open and read the entire input file with UTF-8 encoding to handle special characters
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Split the content into individual lines for processing
    lines = content.split('\n')
    # List to store the cleaned lines that will make up the final output
    cleaned_lines = []
    # Flag to track whether we're currently inside a category wrapper section
    skip_line = False
    
    # Process each line in the input file
    for line in lines:
        # Check if the current line is a category wrapper (either start or end)
        if line.startswith('=== START of ') or line.startswith('=== END of '):
            # Set flag to skip this line and subsequent lines until we hit a separator
            skip_line = True
            continue
        
        # If we're not in a category wrapper section, add the line to our cleaned content
        if not skip_line:
            cleaned_lines.append(line)
        else:
            # If we encounter a separator ("---") while skipping lines,
            # reset the skip flag and include the separator in our output
            if line.strip() == '---':
                skip_line = False
                cleaned_lines.append(line)
    
    # Join all cleaned lines back together with newline characters
    cleaned_content = '\n'.join(cleaned_lines)
    
    # Clean up any excessive empty lines (more than 2 consecutive newlines)
    # This helps maintain consistent formatting in the output
    while '\n\n\n' in cleaned_content:
        cleaned_content = cleaned_content.replace('\n\n\n', '\n\n')
    
    # Write the cleaned content to the output file with UTF-8 encoding
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_content)

if __name__ == '__main__':
    # Import os module for handling file paths
    import os
    
    # Get the absolute path of the directory containing this script
    # This ensures the script works correctly regardless of where it's run from
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct full paths for input and output files by joining the script directory
    # with the respective filenames
    input_file = os.path.join(script_dir, 'knowledge_base_to_chunk.txt')
    output_file = os.path.join(script_dir, 'cleaned_knowledge_base.txt')
    
    # Call the cleaning function with the constructed file paths
    clean_knowledge_base(input_file, output_file)
    # Print confirmation message with the output file path
    print(f"Cleaned knowledge base has been saved to {output_file}")
