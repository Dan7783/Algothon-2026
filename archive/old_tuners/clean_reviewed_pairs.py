import json

# File names
input_file = 'review_decisions.json'
accepted_output_file = 'accepted_pairs.json'
flagged_output_file = 'flagged_pairs.json'

# 1. Load the JSON data from the file
try:
    with open(input_file, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found. Please ensure it is in the same directory as this script.")
    exit()

# 2. Create dictionaries to hold the separated pairs
accepted_pairs = {}
flagged_pairs = {}

# 3. Loop through and sort the pairs based on the "decision" key
for pair, details in data.items():
    decision = details.get("decision", "")
    
    # Capture both "accepted" and "auto accepted"
    if "accepted" in decision:
        accepted_pairs[pair] = details
        
    # Capture "flagged"
    elif decision == "flagged":
        flagged_pairs[pair] = details

# 4. Save the cleaned lists into new JSON files
with open(accepted_output_file, 'w') as f:
    json.dump(accepted_pairs, f, indent=4)

with open(flagged_output_file, 'w') as f:
    json.dump(flagged_pairs, f, indent=4)

# 5. Print a summary to the console
print("Processing complete!")
print(f" - Accepted pairs: {len(accepted_pairs)} (Saved to '{accepted_output_file}')")
print(f" - Flagged pairs:  {len(flagged_pairs)} (Saved to '{flagged_output_file}')")