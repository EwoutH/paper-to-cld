# Step 2: Consolidate and Validate Relations with Goal-Directed Filtering
# pip install google-genai pandas python-dotenv

import os
import glob
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv


def load_consolidation_prompt(goal_description):
    """Load the prompt template for relation consolidation with goal filtering."""
    prompt = f"""
You are an expert in causal analysis and systems thinking. Your task is to consolidate causal relationships from multiple research papers, focusing specifically on relationships relevant to a particular goal.

GOAL/FOCUS: {goal_description}

You will receive multiple text files, each containing:
1. Paper metadata (filename, title, date, authors, DOI, citation)
2. A CSV of causal relationships from that paper

Your task has TWO PARTS:

PART 1: RELEVANT VARIABLE SELECTION
First, identify the key variables that are relevant to the research goal. Select 25-40 most important variables that either:
- Directly relate to the goal/outcome of interest
- Are major drivers or influences in the system
- Participate in important feedback loops
- Are policy-relevant intervention points

PART 2: CONSOLIDATE RELEVANT RELATIONSHIPS
For relationships between the selected variables:
- MERGE similar relationships across papers
- RESOLVE conflicts about polarity (choose most supported)
- STANDARDIZE variable names (use clearest version)
- CITE supporting papers properly

Return your response as a JSON object with this EXACT structure:

{{
  "selected_variables": [
    {{
      "variable_name": "standardized variable name",
      "definition": "clear definition of this variable",
      "unit": "units of measurement (if applicable, otherwise "")",
    }}
  ],
  "consolidated_relationships": [
    {{
      "causal_variable": "standardized cause variable name",
      "effect_variable": "standardized effect variable name", 
      "relationship_name": "descriptive name for relationship",
      "polarity": "positive or negative",
      "supporting_citations": "comma-separated APA citations",
    }}
  ]
}}

IMPORTANT RULES:
- Only include variables and relationships relevant to the goal
- Only include relationships between selected variables
- Use exact APA citations from paper headers
- Ensure JSON is valid (proper quotes, commas, brackets)
- Focus on the most important causal pathways for the goal

SELECTION CRITERIA:
- Focus on variables most relevant to: {goal_description}
- Prioritize relationships with strong empirical support
- Only include direct causal relationships
- Include key feedback loops and leverage points
- Exclude peripheral or weakly-supported relationships
- Merge similar variables (e.g., "GDP growth" and "Economic growth")

IMPORTANT: 
- Return exactly two CSV sections: "PART 1: SELECTED VARIABLES" followed by "PART 2: CONSOLIDATED RELATIONSHIPS"
- Only include relationships between variables from Part 1
- Use exact APA citations from paper headers
- Ensure JSON is valid (proper quotes, commas, brackets)
- Focus on the most important causal pathways for the goal

Here are the extracted relations from all papers:

"""
    return prompt


def load_extraction_files(extraction_dir):
    """Load all extraction text files from Step 1."""
    extraction_files = glob.glob(os.path.join(extraction_dir, "*.txt"))

    if not extraction_files:
        print(f"No extraction files found in {extraction_dir}")
        return ""

    print(f"Found {len(extraction_files)} extraction files")

    combined_content = ""
    for file_path in extraction_files:
        filename = os.path.basename(file_path)
        print(f"Loading: {filename}")

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                combined_content += f"\n\n--- {filename} ---\n"
                combined_content += content
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    return combined_content


def extract_json_from_response(response_text):
    """Extract JSON from LLM response."""
    try:
        # Find JSON content between first { and last }
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}')

        if start_idx == -1 or end_idx == -1:
            print("No JSON object found in response")
            return None

        json_str = response_text[start_idx:end_idx + 1]
        result = json.loads(json_str)

        # Validate required structure
        if 'selected_variables' not in result or 'consolidated_relationships' not in result:
            print("Invalid JSON structure - missing required keys")
            return None

        return result

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON: {e}")
        print(f"Response preview: {response_text[:500]}...")
        return None


def consolidate_with_llm(combined_extractions, goal_description, client, model="gemini-2.5-flash-preview-05-20"):
    """Use LLM to consolidate relations with goal-directed filtering."""
    print("Starting goal-directed consolidation with LLM...")

    # Prepare the full prompt
    prompt = load_consolidation_prompt(goal_description) + combined_extractions

    # Check token length (rough estimate)
    token_estimate = len(prompt) / 4
    print(f"Estimated tokens: {token_estimate:.0f}")

    if token_estimate > 180000:
        print("Warning: Prompt is very long, may exceed model limits")

    try:
        # Create content for the model
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            ),
        ]

        # Process with LLM
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.1),
        )

        response_text = response.text
        print(f"LLM response preview: {response_text[:300]}...")

        # Extract and validate JSON
        result_json = extract_json_from_response(response_text)

        if result_json:
            num_variables = len(result_json.get('selected_variables', []))
            num_relationships = len(result_json.get('consolidated_relationships', []))
            print(f"✓ Successfully consolidated: {num_variables} variables, {num_relationships} relationships")
            return result_json, response_text
        else:
            print("✗ Failed to parse valid JSON from LLM response")
            return None, response_text

    except Exception as e:
        print(f"Error during LLM consolidation: {str(e)}")
        return None, f"Error: {str(e)}"


def validate_and_enhance_result(result_json):
    """Validate and enhance the consolidation result."""
    if not result_json:
        return None

    # Get variable names for validation
    selected_var_names = {var['variable_name'] for var in result_json.get('selected_variables', [])}

    # Validate relationships reference selected variables
    valid_relationships = []
    for rel in result_json.get('consolidated_relationships', []):
        causal_var = rel.get('causal_variable', '')
        effect_var = rel.get('effect_variable', '')

        if causal_var in selected_var_names and effect_var in selected_var_names:
            valid_relationships.append(rel)
        else:
            print(f"Warning: Relationship references unselected variable: {causal_var} -> {effect_var}")

    # Update result with only valid relationships
    result_json['consolidated_relationships'] = valid_relationships

    # Add metadata
    result_json['metadata'] = {
        'total_selected_variables': len(result_json['selected_variables']),
        'total_relationships': len(valid_relationships),
        'positive_relationships': sum(1 for r in valid_relationships if r.get('polarity') == 'positive'),
        'negative_relationships': sum(1 for r in valid_relationships if r.get('polarity') == 'negative'),
        'high_evidence_relationships': sum(1 for r in valid_relationships if r.get('evidence_strength') == 'high')
    }

    return result_json


def create_summary_report(result_json, goal_description, output_dir):
    """Create a human-readable summary report."""
    if not result_json:
        return

    report_path = f"{output_dir}/step2_summary_report.txt"

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("STEP 2 GOAL-DIRECTED CONSOLIDATION REPORT\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"GOAL/FOCUS:\n{goal_description}\n\n")

        metadata = result_json.get('metadata', {})
        f.write("CONSOLIDATION RESULTS:\n")
        f.write(f"  Selected variables: {metadata.get('total_selected_variables', 0)}\n")
        f.write(f"  Consolidated relationships: {metadata.get('total_relationships', 0)}\n")
        f.write(f"  Positive relationships: {metadata.get('positive_relationships', 0)}\n")
        f.write(f"  Negative relationships: {metadata.get('negative_relationships', 0)}\n")
        f.write(f"  High evidence relationships: {metadata.get('high_evidence_relationships', 0)}\n\n")

        f.write("SELECTED VARIABLES:\n")
        for var in result_json.get('selected_variables', []):
            f.write(f"  • {var['variable_name']}\n")
            f.write(f"    Definition: {var['definition']}\n")

        f.write("KEY RELATIONSHIPS:\n")
        for rel in result_json.get('consolidated_relationships', []):
            f.write(f"  • {rel['causal_variable']} → {rel['effect_variable']} ({rel['polarity']})\n")
            f.write(f"    Relationship: {rel['relationship_name']}\n")
            f.write(f"    Sources: {rel['supporting_citations']}\n\n")

    print(f"✓ Summary report saved: {report_path}")


def process_step2(extraction_dir, goal_description, output_dir="results"):
    """
    Step 2: Goal-directed consolidation and validation of extracted relations.

    Args:
        extraction_dir (str): Directory containing Step 1 extraction files
        goal_description (str): Description of the goal/focus for filtering relationships
        output_dir (str): Directory to save results
    """
    # Initialize the Gemini client
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading extractions from: {extraction_dir}")
    print(f"Goal/Focus: {goal_description}")

    # Load all extraction files
    combined_extractions = load_extraction_files(extraction_dir)

    if not combined_extractions:
        print("No valid extractions found")
        return

    # Consolidate using LLM with goal filtering
    result_json, raw_response = consolidate_with_llm(combined_extractions, goal_description, client)

    # Validate and enhance result
    result_json = validate_and_enhance_result(result_json)

    if not result_json:
        print("✗ Consolidation failed")
        return

    # Save raw LLM response
    response_path = f"{output_dir}/step2_raw_response.txt"
    with open(response_path, 'w', encoding='utf-8') as f:
        f.write(raw_response)

    # Save consolidated JSON
    json_path = f"{output_dir}/step2_consolidated_relations.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result_json, f, indent=2, ensure_ascii=False)

    # Create summary report
    create_summary_report(result_json, goal_description, output_dir)

    print(f"\n✓ Step 2 completed successfully!")
    print(f"  - Selected variables: {len(result_json['selected_variables'])}")
    print(f"  - Consolidated relationships: {len(result_json['consolidated_relationships'])}")
    print(f"  - JSON saved: {json_path}")
    print(f"  - Raw response saved: {response_path}")


if __name__ == "__main__":
    # Configuration
    extraction_dir = "results/step1_extractions"

    # Define your research goal and focus areas
    goal_description = """Understand in- or decreases in vehicle distance travelled by all cars when Autonomous Vehicles (AVs) are introduced."""

    focus_areas = """"""

    # Run Step 2
    process_step2(
        extraction_dir=extraction_dir,
        goal_description=goal_description,
        output_dir="results"
    )
