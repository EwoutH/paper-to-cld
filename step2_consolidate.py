# Step 2: Consolidate and Validate Relations with Goal-Oriented Filtering
# pip install google-genai pandas python-dotenv

import os
import glob
import pandas as pd
from google import genai
from google.genai import types
from dotenv import load_dotenv
import io
from pathlib import Path
import json


def load_variable_selection_prompt():
    """Load the prompt template for goal-oriented variable selection."""
    prompt = """
You are an expert in causal analysis and systems thinking. Your task is to identify the most relevant variables for a specific research goal from extracted causal relationships.

RESEARCH GOAL: {research_goal}

FOCUS AREAS: {focus_areas}

You will receive multiple text files containing causal relationships extracted from research papers. Your job is to:

1. IDENTIFY the most relevant variables that directly or indirectly relate to the research goal
2. DEFINE each selected variable clearly 
3. ENSURE the variable set is comprehensive but manageable (aim for 15-30 key variables)

Consider variables that:
- Directly impact the goal/outcome of interest
- Are major drivers or root causes in the system
- Serve as important mediating factors
- Are policy-relevant or actionable
- Create feedback loops with the goal variables

IMPORTANT: Return your results as a CSV with exactly these column headers:
variable_name,variable_definition,relevance_to_goal,variable_type

Do not include any paper metadata, titles, authors, or other text above the CSV. Start your response directly with the header row above, followed by the data rows.

Where:
- variable_name: Standardized name for the variable
- variable_definition: Clear definition of what this variable represents
- relevance_to_goal: Brief explanation of how this variable relates to the research goal
- variable_type: "outcome", "driver", "mediator", or "context"

Do not include any other text, explanations, or markdown formatting. Only return the CSV data with header row.

Here are the extracted relations from all papers:

"""
    return prompt


def load_relation_consolidation_prompt():
    """Load the prompt template for relation consolidation among selected variables."""
    prompt = """
You are an expert in causal analysis and systems thinking. Your task is to consolidate causal relationships ONLY among a pre-selected set of relevant variables.

RESEARCH GOAL: {research_goal}

SELECTED VARIABLES: {selected_variables}

You will receive the same extracted relationships, but now you should ONLY consolidate relationships where BOTH the causal variable AND effect variable are in the selected variables list above.

Your job is to:
1. FILTER relationships to only include those between selected variables
2. MERGE similar relationships (same causal and effect variables but different wording)
3. RESOLVE conflicts between sources about polarity (choose the most supported)
4. STANDARDIZE variable names to match the selected variables list exactly
5. VALIDATE the strength of evidence for each relationship

For each consolidated relationship, provide:
- causal_variable: Must exactly match a name from the selected variables list
- effect_variable: Must exactly match a name from the selected variables list  
- relationship_name: Clear descriptive name for the relationship
- polarity: "positive" or "negative" (consensus across sources)
- supporting_citations: Comma-separated list of APA in-text citations that support this relationship

IMPORTANT: Return ONLY a CSV with exactly these column headers:
causal_variable,effect_variable,relationship_name,polarity,supporting_citations

Do not include any paper metadata, titles, authors, or other text above the CSV. Start your response directly with the header row above, followed by the data rows.

Rules:
- ONLY include relationships where both variables are in the selected variables list
- Use the exact variable names from the selected variables list
- If papers disagree on polarity, choose the one with more/stronger evidence
- Use the exact APA citations provided in the paper headers
- Merge similar relationships between the same variable pairs

Do not include any other text, explanations, or markdown formatting. Only return the CSV data with header row.

Here are the extracted relations from all papers:

"""
    return prompt


def parse_csv_response(response_text):
    """Parse CSV from LLM response."""
    try:
        lines = response_text.strip().split('\n')
        csv_lines = []

        # Find the header row
        header_found = False
        for line in lines:
            # Look for different possible headers
            if ('variable_name' in line or 'causal_variable' in line) and (
                    'variable_definition' in line or 'effect_variable' in line):
                header_found = True
                csv_lines.append(line)
            elif header_found and line.strip() and not line.startswith('---'):
                csv_lines.append(line)
            elif header_found and not line.strip():
                break

        if not csv_lines:
            print("No CSV header found in response")
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
        print(f"Error parsing CSV response: {e}")
        print(f"Response preview: {response_text[:500]}...")
        return pd.DataFrame()


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


def select_relevant_variables(combined_extractions, research_goal, focus_areas, client,
                              model="gemini-2.5-flash-preview-04-17"):
    """Use LLM to select relevant variables based on research goal."""
    print("Step 2a: Selecting relevant variables based on research goal...")

    # Prepare the prompt
    prompt = load_variable_selection_prompt().format(
        research_goal=research_goal,
        focus_areas=focus_areas
    ) + combined_extractions

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
        print(f"Variable selection response preview: {response_text[:300]}...")

        # Parse the variables CSV
        variables_df = parse_csv_response(response_text)

        if not variables_df.empty:
            print(f"✓ Selected {len(variables_df)} relevant variables")
            return variables_df, response_text
        else:
            print("✗ Failed to parse selected variables from LLM response")
            return pd.DataFrame(), response_text

    except Exception as e:
        print(f"Error during variable selection: {str(e)}")
        return pd.DataFrame(), f"Error: {str(e)}"


def consolidate_relations_among_selected(combined_extractions, selected_variables_df, research_goal, client,
                                         model="gemini-2.5-flash-preview-04-17"):
    """Use LLM to consolidate relations among selected variables only."""
    print("Step 2b: Consolidating relations among selected variables...")

    # Create list of selected variable names
    selected_var_names = selected_variables_df['variable_name'].tolist()
    selected_vars_text = "\n".join([f"- {name}: {definition}" for name, definition in
                                    zip(selected_variables_df['variable_name'],
                                        selected_variables_df['variable_definition'])])

    # Prepare the prompt
    prompt = load_relation_consolidation_prompt().format(
        research_goal=research_goal,
        selected_variables=selected_vars_text
    ) + combined_extractions

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
        print(f"Relation consolidation response preview: {response_text[:300]}...")

        # Parse the relations CSV
        relations_df = parse_csv_response(response_text)

        if not relations_df.empty:
            # Validate that all variables are in the selected list
            invalid_causal = ~relations_df['causal_variable'].isin(selected_var_names)
            invalid_effect = ~relations_df['effect_variable'].isin(selected_var_names)

            if invalid_causal.any() or invalid_effect.any():
                print(f"Warning: Some relations use variables not in selected list")
                print(f"Invalid causal variables: {relations_df[invalid_causal]['causal_variable'].unique()}")
                print(f"Invalid effect variables: {relations_df[invalid_effect]['effect_variable'].unique()}")

                # Filter to only valid relations
                valid_relations = ~invalid_causal & ~invalid_effect
                relations_df = relations_df[valid_relations]
                print(f"Filtered to {len(relations_df)} valid relations")

            print(f"✓ Consolidated {len(relations_df)} relations among selected variables")
            return relations_df, response_text
        else:
            print("✗ Failed to parse consolidated relations from LLM response")
            return pd.DataFrame(), response_text

    except Exception as e:
        print(f"Error during relation consolidation: {str(e)}")
        return pd.DataFrame(), f"Error: {str(e)}"


def create_goal_oriented_analysis(variables_df, relations_df, research_goal, focus_areas, output_dir):
    """Create analysis report focused on the research goal."""

    report_path = f"{output_dir}/step2_goal_oriented_analysis.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("GOAL-ORIENTED CAUSAL ANALYSIS REPORT\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"RESEARCH GOAL: {research_goal}\n")
        f.write(f"FOCUS AREAS: {focus_areas}\n\n")

        # Variable analysis by type
        f.write("SELECTED VARIABLES BY TYPE:\n")
        if not variables_df.empty:
            var_types = variables_df['variable_type'].value_counts()
            for var_type, count in var_types.items():
                f.write(f"  {var_type.capitalize()}: {count} variables\n")
            f.write(f"  Total: {len(variables_df)} variables\n\n")

            # List variables by type
            for var_type in ['outcome', 'driver', 'mediator', 'context']:
                type_vars = variables_df[variables_df['variable_type'] == var_type]
                if not type_vars.empty:
                    f.write(f"{var_type.upper()} VARIABLES:\n")
                    for _, row in type_vars.iterrows():
                        f.write(f"  - {row['variable_name']}: {row['variable_definition']}\n")
                        f.write(f"    Relevance: {row['relevance_to_goal']}\n")
                    f.write("\n")

        # Relationship analysis
        if not relations_df.empty:
            f.write("CAUSAL RELATIONSHIPS ANALYSIS:\n")
            total_relations = len(relations_df)
            positive_relations = (relations_df['polarity'] == 'positive').sum()
            negative_relations = (relations_df['polarity'] == 'negative').sum()

            f.write(f"  Total relationships: {total_relations}\n")
            f.write(
                f"  Positive relationships: {positive_relations} ({positive_relations / total_relations * 100:.1f}%)\n")
            f.write(
                f"  Negative relationships: {negative_relations} ({negative_relations / total_relations * 100:.1f}%)\n\n")

            # Most connected variables
            f.write("MOST CONNECTED VARIABLES:\n")
            causal_counts = relations_df['causal_variable'].value_counts()
            effect_counts = relations_df['effect_variable'].value_counts()
            total_counts = causal_counts.add(effect_counts, fill_value=0).sort_values(ascending=False)

            for var, count in total_counts.head(10).items():
                causal_count = causal_counts.get(var, 0)
                effect_count = effect_counts.get(var, 0)
                f.write(
                    f"  - {var}: {int(count)} total connections ({int(causal_count)} outgoing, {int(effect_count)} incoming)\n")
            f.write("\n")

            # Key relationships involving outcome variables
            outcome_vars = variables_df[variables_df['variable_type'] == 'outcome']['variable_name'].tolist()
            if outcome_vars:
                f.write("KEY RELATIONSHIPS INVOLVING OUTCOME VARIABLES:\n")
                outcome_relations = relations_df[
                    relations_df['causal_variable'].isin(outcome_vars) |
                    relations_df['effect_variable'].isin(outcome_vars)
                    ]
                for _, row in outcome_relations.head(15).iterrows():
                    f.write(f"  - {row['causal_variable']} → {row['effect_variable']} ({row['polarity']})\n")
                    f.write(f"    {row['relationship_name']}\n")
                f.write("\n")

        f.write("SYSTEM INSIGHTS FOR GOAL ACHIEVEMENT:\n")
        if not variables_df.empty:
            driver_vars = variables_df[variables_df['variable_type'] == 'driver']['variable_name'].tolist()
            outcome_vars = variables_df[variables_df['variable_type'] == 'outcome']['variable_name'].tolist()

            if driver_vars and outcome_vars:
                f.write("Key leverage points (driver variables that could influence outcomes):\n")
                for driver in driver_vars[:5]:
                    driver_relations = relations_df[relations_df['causal_variable'] == driver]
                    if not driver_relations.empty:
                        affected_outcomes = driver_relations[driver_relations['effect_variable'].isin(outcome_vars)]
                        if not affected_outcomes.empty:
                            f.write(f"  - {driver}: directly influences {len(affected_outcomes)} outcome variable(s)\n")
                        else:
                            f.write(
                                f"  - {driver}: influences {len(driver_relations)} variables (indirect effects possible)\n")

    print(f"✓ Goal-oriented analysis saved: {report_path}")
    return report_path


def process_step2_goal_oriented(extraction_dir, research_goal, focus_areas, output_dir="results"):
    """
    Step 2: Goal-oriented consolidation and validation of relations.

    Args:
        extraction_dir (str): Directory containing Step 1 extraction files
        research_goal (str): Main research question or goal for the CLD
        focus_areas (str): Specific areas of focus or constraints
        output_dir (str): Directory to save results
    """
    # Initialize the Gemini client
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading extractions from: {extraction_dir}")
    print(f"Research goal: {research_goal}")
    print(f"Focus areas: {focus_areas}")

    # Load all extraction files
    combined_extractions = load_extraction_files(extraction_dir)

    if not combined_extractions:
        print("No valid extractions found")
        return

    # Step 2a: Select relevant variables
    variables_df, variables_response = select_relevant_variables(
        combined_extractions, research_goal, focus_areas, client
    )

    if variables_df.empty:
        print("✗ Variable selection failed")
        return

    # Save variable selection results
    variables_csv_path = f"{output_dir}/step2a_selected_variables.csv"
    variables_df.to_csv(variables_csv_path, index=False)

    variables_response_path = f"{output_dir}/step2a_variable_selection_response.txt"
    with open(variables_response_path, 'w', encoding='utf-8') as f:
        f.write(variables_response)

    print(f"✓ Selected variables saved: {variables_csv_path}")

    # Step 2b: Consolidate relations among selected variables
    relations_df, relations_response = consolidate_relations_among_selected(
        combined_extractions, variables_df, research_goal, client
    )

    if relations_df.empty:
        print("✗ Relation consolidation failed")
        return

    # Save relation consolidation results
    relations_csv_path = f"{output_dir}/step2b_consolidated_relations.csv"
    relations_df.to_csv(relations_csv_path, index=False)

    relations_response_path = f"{output_dir}/step2b_relation_consolidation_response.txt"
    with open(relations_response_path, 'w', encoding='utf-8') as f:
        f.write(relations_response)

    # Save Excel version with both sheets
    excel_path = f"{output_dir}/step2_goal_oriented_results.xlsx"
    with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
        variables_df.to_excel(writer, sheet_name='Selected Variables', index=False)
        relations_df.to_excel(writer, sheet_name='Consolidated Relations', index=False)

    # Create goal-oriented analysis report
    analysis_path = create_goal_oriented_analysis(variables_df, relations_df, research_goal, focus_areas, output_dir)

    # Save configuration for reference
    config = {
        'research_goal': research_goal,
        'focus_areas': focus_areas,
        'selected_variables_count': len(variables_df),
        'consolidated_relations_count': len(relations_df),
        'variable_types': variables_df['variable_type'].value_counts().to_dict() if not variables_df.empty else {},
        'relationship_polarities': relations_df['polarity'].value_counts().to_dict() if not relations_df.empty else {}
    }

    config_path = f"{output_dir}/step2_configuration.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    print(f"\n✓ Step 2 completed successfully!")
    print(f"  - Selected variables: {len(variables_df)}")
    print(f"  - Consolidated relations: {len(relations_df)}")
    print(f"  - Variables CSV: {variables_csv_path}")
    print(f"  - Relations CSV: {relations_csv_path}")
    print(f"  - Excel summary: {excel_path}")
    print(f"  - Analysis report: {analysis_path}")


if __name__ == "__main__":
    # Configuration
    extraction_dir = "results/step1_extractions"

    # Define your research goal and focus areas
    research_goal = """Understand in- or decreases in vehicle distance travelled by all cars when Autonomous Vehicles (AVs) are introduced."""

    focus_areas = """"""

    # Run Step 2
    process_step2_goal_oriented(
        extraction_dir=extraction_dir,
        research_goal=research_goal,
        focus_areas=focus_areas,
        output_dir="results"
    )
