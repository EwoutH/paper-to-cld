# Step 2: Consolidate and Validate Relations
# pip install google-genai pandas python-dotenv

import os
import glob
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
import io
from pathlib import Path


def load_consolidation_prompt(goal_description):
    """Load the prompt template for goal-directed relation consolidation."""
    prompt = f"""
You are an expert in causal analysis and systems thinking. Your task is to create a focused, goal-directed Causal Loop Diagram by consolidating and validating causal relationships from multiple research papers.

RESEARCH GOAL/FOCUS: {goal_description}

You will receive multiple text files, each containing:
1. Paper metadata (filename, title, date, authors, DOI, citation)
2. A CSV of causal relationships from that paper

Your task has TWO PARTS - return both parts in a single response:

PART 1: RELEVANT VARIABLE SELECTION
First, identify the key variables that are relevant to the research goal. Select 15-25 most important variables that either:
- Directly relate to the goal/outcome of interest
- Are major drivers or influences in the system
- Participate in important feedback loops
- Are policy-relevant intervention points

Format as CSV with headers: variable_name,definition,relevance_to_goal

PART 2: RELEVANT RELATIONSHIP CONSOLIDATION
Then, consolidate ONLY the relationships between the selected variables. For each relationship:
- causal_variable: Must be from your selected variables list
- effect_variable: Must be from your selected variables list  
- relationship_name: Clear descriptive name
- polarity: "positive" or "negative" (consensus across sources)
- supporting_citations: APA in-text citations

Format as CSV with headers: causal_variable,effect_variable,relationship_name,polarity,supporting_citations

SELECTION CRITERIA:
- Focus on variables most relevant to: {goal_description}
- Prioritize relationships with strong empirical support
- Include key feedback loops and leverage points
- Exclude peripheral or weakly-supported relationships
- Merge similar variables (e.g., "GDP growth" and "Economic growth")

IMPORTANT: 
- Return exactly two CSV sections: "PART 1: SELECTED VARIABLES" followed by "PART 2: CONSOLIDATED RELATIONSHIPS"
- Only include relationships between variables from Part 1
- Use exact APA citations from paper headers
- No other text, explanations, or markdown formatting

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


def parse_two_part_response(response_text):
    """Parse the two-part response: variables and relationships."""
    try:
        lines = response_text.strip().split('\n')

        # Find the two parts
        variables_df = pd.DataFrame()
        relationships_df = pd.DataFrame()

        current_part = None
        csv_lines = []

        for line in lines:
            if 'PART 1' in line and 'SELECTED VARIABLES' in line:
                current_part = 'variables'
                csv_lines = []
            elif 'PART 2' in line and 'CONSOLIDATED RELATIONSHIPS' in line:
                # Process variables CSV if we have it
                if csv_lines and current_part == 'variables':
                    variables_df = parse_csv_from_lines(csv_lines)
                current_part = 'relationships'
                csv_lines = []
            elif current_part and line.strip():
                # Look for CSV headers
                if ('variable_name' in line and current_part == 'variables') or \
                        ('causal_variable' in line and current_part == 'relationships'):
                    csv_lines = [line]
                elif csv_lines and line.strip() and not line.startswith('---'):
                    csv_lines.append(line)
                elif not line.strip() and csv_lines:
                    # End of current CSV
                    if current_part == 'variables' and variables_df.empty:
                        variables_df = parse_csv_from_lines(csv_lines)
                    elif current_part == 'relationships' and relationships_df.empty:
                        relationships_df = parse_csv_from_lines(csv_lines)
                    csv_lines = []

        # Process final CSV if we ended with one
        if csv_lines:
            if current_part == 'variables' and variables_df.empty:
                variables_df = parse_csv_from_lines(csv_lines)
            elif current_part == 'relationships' and relationships_df.empty:
                relationships_df = parse_csv_from_lines(csv_lines)

        return variables_df, relationships_df

    except Exception as e:
        print(f"Error parsing two-part response: {e}")
        print(f"Response preview: {response_text[:500]}...")
        return pd.DataFrame(), pd.DataFrame()


def parse_csv_from_lines(csv_lines):
    """Parse CSV from list of lines."""
    try:
        if not csv_lines:
            return pd.DataFrame()

        csv_content = '\n'.join(csv_lines)
        df = pd.read_csv(io.StringIO(csv_content))

        # Clean up the data
        df = df.dropna(how='all')
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.strip()

        return df

    except Exception as e:
        print(f"Error parsing CSV from lines: {e}")
        return pd.DataFrame()


def consolidate_with_llm(combined_extractions, goal_description, client, model="gemini-2.5-flash-preview-04-17"):
    """Use LLM to consolidate relations focused on a specific goal."""
    print("Starting goal-directed LLM consolidation...")
    print(f"Research goal: {goal_description}")

    # Prepare the full prompt
    prompt = load_consolidation_prompt(goal_description) + combined_extractions

    # Check token length (rough estimate)
    token_estimate = len(prompt) / 4  # Rough approximation
    print(f"Estimated tokens: {token_estimate:.0f}")

    if token_estimate > 100000:  # If too long, we might need to chunk
        print("Warning: Prompt is very long, may need chunking for some models")

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

        # Parse the two-part response
        variables_df, relationships_df = parse_two_part_response(response_text)

        if not variables_df.empty and not relationships_df.empty:
            print(f"✓ Successfully selected {len(variables_df)} variables and {len(relationships_df)} relationships")
            return variables_df, relationships_df, response_text
        else:
            print("✗ Failed to parse variables and/or relationships from LLM response")
            return pd.DataFrame(), pd.DataFrame(), response_text

    except Exception as e:
        print(f"Error during LLM consolidation: {str(e)}")
        return pd.DataFrame(), pd.DataFrame(), f"Error: {str(e)}"


def create_analysis_report(variables_df, relationships_df, goal_description, output_dir):
    """Create a detailed analysis report of the goal-directed consolidation."""

    if variables_df.empty or relationships_df.empty:
        return

    # Count supporting papers for each relation
    relationships_df['num_supporting_papers'] = relationships_df['supporting_citations'].str.count('\(')

    # Analysis statistics
    total_variables = len(variables_df)
    total_relationships = len(relationships_df)
    avg_support = relationships_df['num_supporting_papers'].mean()

    positive_relations = (relationships_df['polarity'] == 'positive').sum()
    negative_relations = (relationships_df['polarity'] == 'negative').sum()

    # Create analysis report
    report_path = f"{output_dir}/step2_goal_directed_analysis.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GOAL-DIRECTED CLD ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"RESEARCH GOAL: {goal_description}\n\n")

        f.write("OVERALL STATISTICS:\n")
        f.write(f"  Selected variables: {total_variables}\n")
        f.write(f"  Consolidated relationships: {total_relationships}\n")
        f.write(f"  Average papers per relationship: {avg_support:.1f}\n")
        f.write(
            f"  Positive relationships: {positive_relations} ({positive_relations / total_relationships * 100:.1f}%)\n")
        f.write(
            f"  Negative relationships: {negative_relations} ({negative_relations / total_relationships * 100:.1f}%)\n\n")

        f.write("SELECTED VARIABLES AND DEFINITIONS:\n")
        for _, row in variables_df.iterrows():
            f.write(f"  {row['variable_name']}: {row['definition']}\n")
            f.write(f"    Relevance: {row['relevance_to_goal']}\n\n")

        f.write("RELATIONSHIP ANALYSIS:\n")
        f.write("Most frequently appearing causal variables:\n")
        causal_counts = relationships_df['causal_variable'].value_counts().head(10)
        for var, count in causal_counts.items():
            f.write(f"  {var}: {count} outgoing relationships\n")
        f.write("\n")

        f.write("Most frequently appearing effect variables:\n")
        effect_counts = relationships_df['effect_variable'].value_counts().head(10)
        for var, count in effect_counts.items():
            f.write(f"  {var}: {count} incoming relationships\n")
        f.write("\n")

        f.write("RELATIONSHIPS BY EVIDENCE STRENGTH:\n")
        support_counts = relationships_df['num_supporting_papers'].value_counts().sort_index()
        for papers, count in support_counts.items():
            f.write(f"  {papers} supporting paper(s): {count} relationships\n")
        f.write("\n")

        f.write("STRONGEST SUPPORTED RELATIONSHIPS (2+ papers):\n")
        strong_relations = relationships_df[relationships_df['num_supporting_papers'] >= 2].sort_values(
            'num_supporting_papers', ascending=False)
        for _, row in strong_relations.head(15).iterrows():
            f.write(
                f"  {row['causal_variable']} → {row['effect_variable']} ({row['polarity']}) - {row['num_supporting_papers']} papers\n")
            f.write(f"    Citations: {row['supporting_citations']}\n")
        f.write("\n")

        f.write("GOAL RELEVANCE ANALYSIS:\n")
        f.write("Variables directly related to the research goal:\n")
        direct_vars = variables_df[variables_df['relevance_to_goal'].str.contains('direct', case=False, na=False)]
        for _, row in direct_vars.iterrows():
            f.write(f"  {row['variable_name']}: {row['relevance_to_goal']}\n")

    print(f"✓ Goal-directed analysis report saved: {report_path}")


def process_step2(extraction_dir, research_goal, output_dir="results"):
    """
    Step 2: Goal-directed consolidation and validation of extracted relations.

    Args:
        extraction_dir (str): Directory containing Step 1 extraction files
        research_goal (str): Research goal/focus for filtering variables and relationships
        output_dir (str): Directory to save results
    """
    # Initialize the Gemini client
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading extractions from: {extraction_dir}")
    print(f"Research goal: {research_goal}")

    # Load all extraction files
    combined_extractions = load_extraction_files(extraction_dir)

    if not combined_extractions:
        print("No valid extractions found")
        return

    # Consolidate using LLM with goal direction
    variables_df, relationships_df, raw_response = consolidate_with_llm(combined_extractions, research_goal, client)

    if variables_df.empty or relationships_df.empty:
        print("✗ Goal-directed consolidation failed")
        return

    # Save raw LLM response
    response_path = f"{output_dir}/step2_raw_response.txt"
    with open(response_path, 'w', encoding='utf-8') as f:
        f.write(f"RESEARCH GOAL: {research_goal}\n\n")
        f.write("=" * 50 + "\n\n")
        f.write(raw_response)

    # Save selected variables
    variables_path = f"{output_dir}/step2_selected_variables.csv"
    variables_df.to_csv(variables_path, index=False)

    variables_excel_path = f"{output_dir}/step2_selected_variables.xlsx"
    variables_df.to_excel(variables_excel_path, index=False)

    # Save consolidated relationships
    relationships_path = f"{output_dir}/step2_consolidated_relations.csv"
    relationships_df.to_csv(relationships_path, index=False)

    relationships_excel_path = f"{output_dir}/step2_consolidated_relations.xlsx"
    relationships_df.to_excel(relationships_excel_path, index=False)

    # Save combined Excel with both sheets
    combined_excel_path = f"{output_dir}/step2_goal_directed_cld.xlsx"
    with pd.ExcelWriter(combined_excel_path, engine='openpyxl') as writer:
        variables_df.to_excel(writer, sheet_name='Selected Variables', index=False)
        relationships_df.to_excel(writer, sheet_name='Consolidated Relations', index=False)

        # Add goal description sheet
        goal_df = pd.DataFrame({'Research Goal': [research_goal]})
        goal_df.to_excel(writer, sheet_name='Research Goal', index=False)

    # Create analysis report
    create_analysis_report(variables_df, relationships_df, research_goal, output_dir)

    print(f"\n✓ Step 2 completed successfully!")
    print(f"  - Selected variables: {len(variables_df)}")
    print(f"  - Consolidated relationships: {len(relationships_df)}")
    print(f"  - Variables saved: {variables_path}")
    print(f"  - Relationships saved: {relationships_path}")
    print(f"  - Combined Excel: {combined_excel_path}")
    print(f"  - Raw response: {response_path}")


if __name__ == "__main__":
    # Configuration
    extraction_dir = "results/step1_extractions"

    # Define your research goal and focus areas
    research_goal = """Understand in- or decreases in vehicle distance travelled by all cars when Autonomous Vehicles (AVs) are introduced."""

    focus_areas = """"""

    # Run Step 2
    process_step2(
        extraction_dir=extraction_dir,
        research_goal=research_goal,
        output_dir="results"
    )
