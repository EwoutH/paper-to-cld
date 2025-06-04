# Step 1: Extract Relations from Papers
# pip install google-genai python-dotenv

import os
import time
import glob
from google import genai
from google.genai import types
from dotenv import load_dotenv


def load_extraction_prompt():
    """Load the prompt template for relation extraction."""
    prompt = """
You are an expert in causal analysis and systems thinking. Your task is to:

1. FIRST, extract the paper metadata and format it exactly as shown below
2. THEN, extract direct causal relationships from the paper

STEP 1 - METADATA EXTRACTION:
Extract and format the following information exactly as shown:

TITLE: [paper title]
DATE: [publication date in YYYY-MM-DD format, use YYYY-01-01 if only year available]
AUTHORS: [author names separated by semicolons]
DOI: [DOI if available, otherwise "Not available"]
CITATION: [APA in-text citation format: (Author, Year) or (Author1 & Author2, Year)]

STEP 2 - RELATION EXTRACTION:
For each direct causal relationship you identify, extract in CSV format with these exact headers:
causal_variable,effect_variable,relationship_name,polarity,context_evidence

Guidelines for extraction:
- causal_variable: The cause or independent variable
- effect_variable: The effect or dependent variable  
- relationship_name: A brief descriptive name for the relationship
- polarity: "positive" (cause increases effect) or "negative" (cause decreases effect)
- context_evidence: Brief quote or context from the paper (keep under 100 characters)

ONLY extract relationships that are EXPLICITLY stated as causal, including:
- Direct causal claims using words like "causes", "leads to", "results in", "drives", "influences"
- Experimental results showing cause-and-effect relationships
- Theoretical models that explicitly specify causal mechanisms
- Statistical analyses that establish causality (not just correlation)

DO NOT extract:
- Simple correlations or associations without causal language
- Implicit or inferred relationships
- Relationships that are only suggested or hypothesized
- Statistical relationships that don't establish causation

IMPORTANT: 
- Start with the 6 metadata lines exactly as formatted above
- Follow with a blank line
- Then provide the CSV with header row
- No other text, explanations, or markdown formatting
"""
    return prompt


def extract_from_paper(pdf_path, client, model="gemini-2.5-flash-preview-05-20"):
    """Extract metadata and causal relations from a single paper."""
    filename = os.path.basename(pdf_path)

    print(f"Processing: {filename}")

    try:
        # Upload the file
        uploaded_file = client.files.upload(file=pdf_path)

        # Create content for the model
        prompt = load_extraction_prompt()
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                    types.Part.from_uri(
                        file_uri=uploaded_file.uri,
                        mime_type=uploaded_file.mime_type,
                    ),
                ],
            ),
        ]

        # Process the file
        response = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(temperature=0.0),
        )

        response_text = response.text
        print(f"Response ({len(response_text)} characters): {response_text[:500]}...")
        response_text = f"FILENAME: {filename}\n{response_text}"

        return response_text

    except Exception as e:
        print(f"Error processing {filename}: {str(e)}")
        return f"Error processing {filename}: {str(e)}"


def process_papers_step1(folder_path, max_papers=None, output_dir="results"):
    """
    Step 1: Extract causal relations from all papers in the folder.

    Args:
        folder_path (str): Path to folder containing PDF files
        max_papers (int, optional): Maximum number of papers to process
        output_dir (str): Directory to save results
    """
    # Initialize the Gemini client
    load_dotenv()
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f"{output_dir}/step1_extractions", exist_ok=True)

    # Find all PDF files
    pdf_files = glob.glob(os.path.join(folder_path, "*.pdf"))
    if not pdf_files:
        print(f"No PDF files found in {folder_path}")
        return

    # Limit number of papers if specified
    if max_papers:
        pdf_files = pdf_files[:max_papers]

    print(f"Found {len(pdf_files)} PDF files to process for relation extraction.")

    successful_extractions = []
    failed_papers = []

    # Process each PDF file
    for i, pdf_file in enumerate(pdf_files):
        filename = os.path.basename(pdf_file)
        paper_name = os.path.splitext(filename)[0]

        # First, check if a txt file already exists
        txt_path = f"{output_dir}/step1_extractions/{paper_name}.txt"
        if os.path.exists(txt_path):
            print(f"Skipping {filename}, extraction already exists at {txt_path}")
            successful_extractions.append(filename)
            continue

        print(f"\n--- Processing file {i + 1}/{len(pdf_files)}: {filename} ---")

        # Extract from this paper
        extraction_result = extract_from_paper(pdf_file, client)

        # Save result as text file
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(extraction_result)

        if "Error processing" not in extraction_result:
            successful_extractions.append(filename)
            print(f"✓ Saved extraction: {txt_path}")
        else:
            failed_papers.append(filename)
            print(f"✗ Failed to extract from {filename}")

        # Rate limiting
        if i < len(pdf_files) - 1:
            print(f"Waiting 7 seconds before next file...")
            time.sleep(7)

    # Save summary
    summary_path = f"{output_dir}/step1_summary.txt"
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("STEP 1 EXTRACTION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total papers processed: {len(pdf_files)}\n")
        f.write(f"Successful extractions: {len(successful_extractions)}\n")
        f.write(f"Failed extractions: {len(failed_papers)}\n\n")

        if successful_extractions:
            f.write("SUCCESSFUL EXTRACTIONS:\n")
            for paper in successful_extractions:
                f.write(f"  - {paper}\n")
            f.write("\n")

        if failed_papers:
            f.write("FAILED EXTRACTIONS:\n")
            for paper in failed_papers:
                f.write(f"  - {paper}\n")

    print(f"\n✓ Step 1 completed!")
    print(f"  - Processed: {len(pdf_files)} papers")
    print(f"  - Successful: {len(successful_extractions)}")
    print(f"  - Failed: {len(failed_papers)}")
    print(f"  - Summary saved: {summary_path}")


if __name__ == "__main__":
    # Configuration
    folder_path = r"C:\Users\Ewout\Documents\KiM - Ewout\Zelfrijdend vervoer\Literatuur\Systematic literature reviews"

    # Run Step 1
    process_papers_step1(
        folder_path=folder_path,
        max_papers=None,  # Process all papers (set to number for testing)
        output_dir="results"
    )
