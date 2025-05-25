# generate_product_descriptions.py

import os
import json
import re # Import regex for cleaning HTML tags

# --- Configuration (MAKE SURE THIS PATH IS CORRECT) ---
# This should point to the directory containing individual product JSON files (e.g., 10001.json, 10002.json)
STYLE_JSON_PATH = "C:/Users/Taher/Downloads/hackathon/fashion-dataset/styles" 

# This is the output file that will be created
PRODUCT_DESCRIPTIONS_FILE = "product_descriptions.json"

def clean_html_tags(text):
    """Removes HTML tags like <p> and <br /> from a string."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text).strip()

def generate_descriptions():
    product_descriptions = {}
    processed_count = 0
    skipped_count = 0
    print(f"Starting to generate product descriptions from: {STYLE_JSON_PATH}")

    if not os.path.isdir(STYLE_JSON_PATH):
        print(f"Error: Directory not found at {STYLE_JSON_PATH}")
        print("Please ensure STYLE_JSON_PATH points to the correct location of your 'styles' folder.")
        return

    for filename in os.listdir(STYLE_JSON_PATH):
        if filename.endswith(".json"):
            # Assuming filename is like "12345.json"
            product_id = filename.replace(".json", "")
            json_path = os.path.join(STYLE_JSON_PATH, filename)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    full_json_data = json.load(f)
                    
                    # Access the 'data' key first
                    data = full_json_data.get("data", {})
                    
                    description_parts = []

                    # 1. Product Display Name (often a good starting point)
                    if data.get("productDisplayName"):
                        description_parts.append(data["productDisplayName"])
                    
                    # 2. Brand Name
                    if data.get("brandName"):
                        description_parts.append(data["brandName"])
                        
                    # 3. Primary description from 'productDescriptors'
                    if "productDescriptors" in data:
                        descriptors = data["productDescriptors"]
                        
                        if "description" in descriptors and descriptors["description"].get("value"):
                            clean_desc = clean_html_tags(descriptors["description"]["value"])
                            if clean_desc:
                                description_parts.append(clean_desc)
                        
                        # 4. Style Note (adds more context)
                        if "style_note" in descriptors and descriptors["style_note"].get("value"):
                            clean_style_note = clean_html_tags(descriptors["style_note"]["value"])
                            if clean_style_note:
                                description_parts.append("Style note: " + clean_style_note)
                                
                    # 5. Key attributes from the top level 'data'
                    if data.get("baseColour") and data["baseColour"] != "NA":
                        description_parts.append(data["baseColour"])
                    if data.get("gender"):
                        description_parts.append(data["gender"])
                    if data.get("masterCategory", {}).get("typeName"):
                        description_parts.append(data["masterCategory"]["typeName"])
                    if data.get("subCategory", {}).get("typeName"):
                        description_parts.append(data["subCategory"]["typeName"])
                    if data.get("articleType", {}).get("typeName"):
                        description_parts.append(data["articleType"]["typeName"])
                    if data.get("season"):
                        description_parts.append(data["season"])
                    if data.get("usage"):
                        description_parts.append(data["usage"])

                    # 6. Attributes from 'articleAttributes' (loop through them)
                    if "articleAttributes" in data:
                        for attr_name, attr_value in data["articleAttributes"].items():
                            # Filter out generic/less useful attributes or "NA" values
                            if attr_value and attr_value != "NA" and attr_name not in ["Multipack Set", "Body or Garment Size", "Number of Pockets", "Surface Styling", "Pattern Coverage"]:
                                description_parts.append(f"{attr_name}: {attr_value}")

                    # Combine all parts into a single string
                    full_description = ". ".join(filter(None, description_parts)).strip()
                    
                    if not full_description:
                        full_description = f"Product ID {product_id}: No detailed description could be extracted."
                        print(f"Warning: No detailed description found for product ID {product_id}. Using fallback.")
                        skipped_count += 1
                    else:
                        processed_count += 1

                    product_descriptions[product_id] = full_description

            except json.JSONDecodeError:
                print(f"Error: Could not decode JSON from {json_path}. Skipping.")
                skipped_count += 1
            except FileNotFoundError:
                print(f"Error: File not found {json_path}. Skipping.")
                skipped_count += 1
            except Exception as e:
                print(f"An unexpected error occurred processing {json_path}: {e}. Skipping.")
                skipped_count += 1

    # Save the generated descriptions to a JSON file
    try:
        with open(PRODUCT_DESCRIPTIONS_FILE, 'w', encoding='utf-8') as f:
            json.dump(product_descriptions, f, indent=4, ensure_ascii=False)
        print(f"\nSuccessfully generated {processed_count} product descriptions.")
        if skipped_count > 0:
            print(f"Skipped {skipped_count} files due to errors or missing data.")
        print(f"Output saved to: {PRODUCT_DESCRIPTIONS_FILE}")
    except Exception as e:
        print(f"Error saving product descriptions to file: {e}")

if __name__ == "__main__":
    generate_descriptions()