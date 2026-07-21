#!/usr/bin/env python3
"""
Convert WikiText JSON files to plain text format for transformer training.

The script processes JSON files in the wikitext-103-temp directory and converts
them to .txt files by extracting text content from the nested JSON structure.
"""

import json
import os
from pathlib import Path


def convert_json_to_txt(json_path, output_path):
    """
    Convert a single JSON file to text format.
    
    Args:
        json_path: Path to input JSON file
        output_path: Path to output text file
    """
    print(f"Processing: {json_path}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract text entries from the nested structure
        if 'text' in data and isinstance(data['text'], dict):
            text_entries = data['text']
            
            # Sort by key (assuming numeric indices)
            sorted_keys = sorted(text_entries.keys(), key=lambda x: int(x))
            
            # Write all text entries to output file
            with open(output_path, 'w', encoding='utf-8') as out:
                for i, key in enumerate(sorted_keys):
                    text = text_entries[key]
                    out.write(text)
                    # Add double newline between entries for separation
                    if i < len(sorted_keys) - 1:
                        out.write('\n\n')
            
            print(f"  ✓ Wrote {len(sorted_keys)} text entries to {output_path}")
            return True
        else:
            print(f"  ✗ Unexpected JSON structure in {json_path}")
            return False
            
    except Exception as e:
        print(f"  ✗ Error processing {json_path}: {e}")
        return False


def main():
    """Main conversion process."""
    # Set up paths
    script_dir = Path(__file__).parent
    wiki_dir = script_dir / "wikitext-103-temp"
    output_dir = script_dir / "wikitext-103-txt"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("WikiText JSON to TXT Converter")
    print("=" * 60)
    print()
    
    # Find all JSON files in the wiki directory
    json_files = sorted(wiki_dir.glob("*.json"))
    
    if not json_files:
        print(f"No JSON files found in {wiki_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to process\n")
    
    # Process each JSON file
    successful = 0
    failed = 0
    
    for json_file in json_files:
        # Create output filename
        output_file = output_dir / json_file.name.replace('.json', '.txt')
        
        if convert_json_to_txt(json_file, output_file):
            successful += 1
        else:
            failed += 1
        print()
    
    # Print summary
    print("=" * 60)
    print(f"Conversion complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

