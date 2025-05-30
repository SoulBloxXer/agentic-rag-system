def clean_line_breakers(input_file, output_file):
    try:
        # Read the input file
        with open(input_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace literal '\n' with actual newlines
        cleaned_content = content.replace('\\n', '\n')
        
        # Write the cleaned content to output file
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(cleaned_content)
            
        print(f"Successfully cleaned line breakers. Output saved to {output_file}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    input_file = "text_file_to_parse.txt"
    output_file = "cleaned_articles.txt"
    clean_line_breakers(input_file, output_file)
