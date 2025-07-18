import streamlit as st
import os
import json
import re
import io
import base64
from typing import Dict, List, Tuple, Optional
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Configure Streamlit page
st.set_page_config(
    page_title="Interactive CLD Builder",
    page_icon="🔄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Load environment variables
load_dotenv()

# Initialize session state
if 'extracted_papers' not in st.session_state:
    st.session_state.extracted_papers = []
if 'selected_variables' not in st.session_state:
    st.session_state.selected_variables = []
if 'variable_selections' not in st.session_state:
    st.session_state.variable_selections = {}
if 'consolidated_relations' not in st.session_state:
    st.session_state.consolidated_relations = []
if 'relation_selections' not in st.session_state:
    st.session_state.relation_selections = {}
if 'goal_description' not in st.session_state:
    st.session_state.goal_description = ""
if 'variable_topics' not in st.session_state:
    st.session_state.variable_topics = []
if 'topic_selections' not in st.session_state:
    st.session_state.topic_selections = {}

class CLDBuilder:
    """Class to build Causal Loop Diagrams interactively."""

    def __init__(self):
        self.client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        self.colors = {
            'positive': '#2E8B57',
            'negative': '#DC143C',
            'node': '#4A90E2',
            'node_border': '#2C5AA0'
        }

    def parse_extraction_file(self, file_content: str) -> Dict:
        """Parse a single extraction file."""
        lines = file_content.strip().split('\n')

        # Extract metadata (first 6 lines)
        metadata = {}
        for i, line in enumerate(lines[:6]):
            if ':' in line:
                key, value = line.split(':', 1)
                metadata[key.strip().lower()] = value.strip()

        # Find CSV data (after blank line)
        csv_start = -1
        for i, line in enumerate(lines):
            if line.strip() == '' and i < len(lines) - 1:
                if 'causal_variable' in lines[i + 1]:
                    csv_start = i + 1
                    break

        relations = []
        if csv_start > 0:
            csv_lines = lines[csv_start:]
            if len(csv_lines) > 1:  # Has header + data
                header = csv_lines[0].split(',')
                for line in csv_lines[1:]:
                    if line.strip():
                        values = [v.strip() for v in line.split(',')]
                        if len(values) >= 5:
                            relations.append({
                                'causal_variable': values[0],
                                'effect_variable': values[1],
                                'relationship_name': values[2],
                                'polarity': values[3],
                                'context_evidence': values[4]
                            })

        return {
            'metadata': metadata,
            'relations': relations
        }

    def get_variable_consolidation_prompt(self, goal_description: str, papers_data: List[Dict]) -> str:
        """Create prompt for variable consolidation."""
        prompt = f"""
You are an expert in causal analysis. Your task is to identify the most important variables for a specific research goal, organized by thematic topics.

GOAL: {goal_description}

You will receive data from multiple research papers. For each paper, you'll see:
1. Paper metadata
2. Extracted causal relationships

Your task: Select 25-60 most important variables that are relevant to the research goal, organized into thematic topics.

Selection criteria:
- Variables directly related to the goal
- Variables are specific, have a clear definition and optionally a unit of measurement
- Major system drivers or influences  
- Variables in important feedback loops
- Policy-relevant intervention points

Return a JSON object with this EXACT structure:
{{
  "variable_topics": [
    {{
      "topic_name": "clear topic name (e.g., Economic Factors, Technology Adoption)",
      "topic_description": "brief description of what this topic covers",
      "variables": [
        {{
          "variable_name": "standardized variable name",
          "definition": "clear definition of this variable",
          "unit": "units of measurement (if applicable, otherwise empty string)"
        }}
      ]
    }}
  ]
}}

Here is the extracted data from all papers:

"""

        # Add paper data
        for paper in papers_data:
            metadata = paper['metadata']
            prompt += f"\n--- Paper: {metadata.get('title', 'Unknown')} ---\n"
            prompt += f"Citation: {metadata.get('citation', 'Unknown')}\n"
            prompt += f"Authors: {metadata.get('authors', 'Unknown')}\n\n"

            prompt += "Causal relationships:\n"
            for rel in paper['relations']:
                prompt += f"  {rel['causal_variable']} → {rel['effect_variable']} ({rel['polarity']})\n"
            prompt += "\n"

        return prompt

    def get_relation_consolidation_prompt(self, goal_description: str, selected_variables: List[str],
                                          papers_data: List[Dict]) -> str:
        """Create prompt for relation consolidation."""
        # Organize selected variables by their topics for better context
        selected_vars_by_topic = {}
        for topic in st.session_state.variable_topics:
            topic_vars = [var['variable_name'] for var in topic['variables']
                          if var['variable_name'] in selected_variables]
            if topic_vars:
                selected_vars_by_topic[topic['topic_name']] = topic_vars

        prompt = f"""
You are an expert in causal analysis. Your task is to consolidate causal relationships between selected variables.

GOAL: {goal_description}

SELECTED VARIABLES BY TOPIC:
"""
        # Add selected variables organized by topic
        for topic_name, vars_in_topic in selected_vars_by_topic.items():
            prompt += f"\n{topic_name}:\n"
            for var in vars_in_topic:
                prompt += f"  - {var}\n"

        prompt += f"""

Your task: For relationships between the selected variables:
- MERGE similar relationships across papers
- RESOLVE conflicts about polarity (choose most supported)
- STANDARDIZE variable names to match the selected variables list above
- CITE supporting papers properly
- Focus on relationships that cross topic boundaries as well as within topics

Return a JSON object with this EXACT structure:
{{
  "consolidated_relationships": [
    {{
      "causal_variable": "standardized cause variable name from selected list",
      "effect_variable": "standardized effect variable name from selected list", 
      "relationship_name": "descriptive name for relationship",
      "polarity": "positive or negative",
      "supporting_citations": "comma-separated APA citations",
      "strength": "high, medium, or low",
      "cross_topic": true/false (indicates if this relationship crosses topic boundaries)
    }}
  ]
}}

IMPORTANT:
- Only include relationships between the selected variables listed above.
- Only one relationship per pair of variables is allowed (no duplicates).
- Only include direct causal relationships.
- Pay special attention to cross-topic relationships as these often reveal important system dynamics.

Here is the extracted data from all papers:

"""

        # Add all paper data - let LLM do the filtering and mapping
        for paper in papers_data:
            metadata = paper['metadata']
            prompt += f"\n--- Paper: {metadata.get('title', 'Unknown')} ---\n"
            prompt += f"Citation: {metadata.get('citation', 'Unknown')}\n"

            prompt += "Causal relationships:\n"
            for rel in paper['relations']:
                if rel['causal_variable'] in selected_variables and rel['effect_variable'] in selected_variables:
                    prompt += f"  {rel['causal_variable']} → {rel['effect_variable']} ({rel['polarity']}): {rel['context_evidence']}\n"
            prompt += "\n"

        print(f"Prompt has {len(prompt.splitlines())} lines and {len(prompt)} characters")
        return prompt

    def call_llm(self, prompt: str) -> Tuple[Optional[Dict], str]:
        """Call LLM and parse JSON response."""
        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=prompt)],
                ),
            ]

            response = self.client.models.generate_content(
                model="gemini-2.5-flash-preview-05-20",
                contents=contents,
                config=types.GenerateContentConfig(temperature=0.1),
            )

            response_text = response.text
            print(f"Response text: {response_text}")

            # Extract JSON
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}')

            if start_idx == -1 or end_idx == -1:
                return None, response_text

            json_str = response_text[start_idx:end_idx + 1]
            result = json.loads(json_str)

            return result, response_text

        except json.JSONDecodeError as e:
            return None, f"JSON Error: {str(e)}"
        except Exception as e:
            return None, f"Error: {str(e)}"

    def create_cld_diagram(self, relations: List[Dict], layout_type: str = 'Kamada-Kawai') -> go.Figure:
        """Create interactive Plotly CLD diagram."""
        # Build graph
        G = nx.DiGraph()

        # Add nodes and edges
        for rel in relations:
            G.add_edge(
                rel['causal_variable'],
                rel['effect_variable'],
                polarity=rel['polarity'],
                relationship_name=rel['relationship_name'],
                citations=rel['supporting_citations']
            )

        if not G.nodes():
            return None

        # Calculate layout
        if layout_type == 'Spring':
            pos = nx.spring_layout(G, k=1, iterations=50, seed=42)
        elif layout_type == 'Circular':
            pos = nx.circular_layout(G)
        elif layout_type == 'Shell':
            pos = nx.shell_layout(G)
        elif layout_type == 'Kamada-Kawai':
            pos = nx.kamada_kawai_layout(G)
        elif layout_type == 'Spectral':
            pos = nx.spectral_layout(G)
        else:
            pos = nx.spring_layout(G, seed=42)

        # Prepare node traces
        node_x, node_y, node_text, node_info, node_sizes = [], [], [], [], []

        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)

            # Shortened display text
            display_text = node if len(node) <= 20 else node[:17] + "..."
            node_text.append(display_text)

            # Hover info
            in_degree = G.in_degree(node)
            out_degree = G.out_degree(node)
            hover_text = f"<b>{node}</b><br>Incoming: {in_degree} | Outgoing: {out_degree}"
            node_info.append(hover_text)

            # Node size based on degree
            degree_centrality = nx.degree_centrality(G)[node]
            node_sizes.append(20 + degree_centrality * 40)

        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            hoverinfo='text',
            hovertext=node_info,
            marker=dict(
                size=node_sizes,
                color=self.colors['node'],
                line=dict(width=2, color=self.colors['node_border'])
            ),
            textfont=dict(size=9, color='white'),
            name='Variables'
        )

        # Prepare edge traces by polarity
        edge_traces = []

        for polarity, color in [('positive', self.colors['positive']), ('negative', self.colors['negative'])]:
            edge_x, edge_y = [], []

            for edge in G.edges(data=True):
                if edge[2]['polarity'] == polarity:
                    x0, y0 = pos[edge[0]]
                    x1, y1 = pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            if edge_x:
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color=color),
                    hoverinfo='none',
                    mode='lines',
                    name=f'{polarity.capitalize()} relationships',
                    showlegend=True
                )
                edge_traces.append(edge_trace)

        # Create figure
        fig = go.Figure(
            data=edge_traces + [node_trace],
            layout=go.Layout(
                title=dict(text='Causal Loop Diagram', x=0.5, font=dict(size=18)),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=60),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                height=600
            )
        )

        return fig

    def detect_feedback_loops(self, relations: List[Dict]) -> List[Dict]:
        """Detect feedback loops in the relationships."""
        G = nx.DiGraph()

        for rel in relations:
            G.add_edge(rel['causal_variable'], rel['effect_variable'], polarity=rel['polarity'])

        cycles = list(nx.simple_cycles(G))

        feedback_loops = []
        for i, cycle in enumerate(cycles):
            if len(cycle) <= 8:  # Reasonable cycle length
                # Determine loop type
                polarities = []
                for j in range(len(cycle)):
                    start = cycle[j]
                    end = cycle[(j + 1) % len(cycle)]
                    if G.has_edge(start, end):
                        polarities.append(G[start][end]['polarity'])

                neg_count = polarities.count('negative')
                loop_type = 'reinforcing' if neg_count % 2 == 0 else 'balancing'

                feedback_loops.append({
                    'cycle_id': i + 1,
                    'nodes': cycle,
                    'length': len(cycle),
                    'loop_type': loop_type,
                    'path': ' → '.join(cycle + [cycle[0]])
                })

        return feedback_loops


def main():
    """Main Streamlit app."""
    st.title("🔄 Interactive Causal Loop Diagram Builder")
    st.markdown("Build CLDs from extracted research paper data with iterative refinement.")

    cld_builder = CLDBuilder()

    # Check API key
    if not os.environ.get("GEMINI_API_KEY"):
        st.error("⚠️ GEMINI_API_KEY not found in environment variables. Please set it to use this app.")
        st.stop()

    # Step 1: File Upload
    st.header("📁 Step 1: Load Extraction Files")
    st.markdown("Upload the .txt files from Step 1 extraction (each contains paper metadata + CSV relations).")

    uploaded_files = st.file_uploader(
        "Choose extraction files",
        type=['txt'],
        accept_multiple_files=True,
        key="extraction_files"
    )

    if uploaded_files:
        if st.button("Load Files", key="load_files"):
            with st.spinner("Loading and parsing files..."):
                extracted_papers = []
                for file in uploaded_files:
                    content = file.read().decode('utf-8')
                    parsed_data = cld_builder.parse_extraction_file(content)
                    parsed_data['filename'] = file.name
                    extracted_papers.append(parsed_data)

                st.session_state.extracted_papers = extracted_papers

        if st.session_state.extracted_papers:
            st.success(f"✅ Loaded {len(st.session_state.extracted_papers)} papers")

            # Show summary
            total_relations = sum(len(paper['relations']) for paper in st.session_state.extracted_papers)
            st.info(f"Total extracted relationships: {total_relations}")

            with st.expander("View loaded papers"):
                for paper in st.session_state.extracted_papers:
                    metadata = paper['metadata']
                    st.write(f"**{metadata.get('title', 'Unknown')}**")
                    st.write(f"Authors: {metadata.get('authors', 'Unknown')}")
                    st.write(f"Relations: {len(paper['relations'])}")
                    st.write("---")

    # Step 2: Goal-Directed Variable Selection
    if st.session_state.extracted_papers:
        st.header("🎯 Step 2: Goal-Directed Variable Selection")

        goal_description = st.text_area(
            "Describe your research goal/focus:",
            value=st.session_state.goal_description,
            height=100,
            help="Describe what you want to understand or the outcome you're interested in."
        )

        if goal_description != st.session_state.goal_description:
            st.session_state.goal_description = goal_description
            # Reset downstream selections when goal changes
            st.session_state.variable_topics = []
            st.session_state.topic_selections = {}
            st.session_state.variable_selections = {}
            st.session_state.consolidated_relations = []
            st.session_state.relation_selections = {}

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("🔍 Consolidate Variables", key="consolidate_vars", disabled=not goal_description.strip()):
                with st.spinner("Calling LLM to select relevant variables..."):
                    prompt = cld_builder.get_variable_consolidation_prompt(goal_description,
                                                                           st.session_state.extracted_papers)
                    result, raw_response = cld_builder.call_llm(prompt)

                    if result and 'variable_topics' in result:
                        st.session_state.variable_topics = result['variable_topics']
                        # Initialize all topics and variables as selected
                        st.session_state.topic_selections = {
                            topic['topic_name']: True for topic in result['variable_topics']
                        }
                        st.session_state.variable_selections = {}
                        for topic in result['variable_topics']:
                            for var in topic['variables']:
                                st.session_state.variable_selections[var['variable_name']] = True

                        # Reset relations when variables change
                        st.session_state.consolidated_relations = []
                        st.session_state.relation_selections = {}

                        total_vars = sum(len(topic['variables']) for topic in result['variable_topics'])
                        st.success(f"✅ Selected {total_vars} variables in {len(result['variable_topics'])} topics")
                    else:
                        st.error("❌ Failed to get valid response from LLM")
                        with st.expander("View raw response"):
                            st.text(raw_response)

        # Show selected variables organized by topics with hierarchical checkboxes
        if st.session_state.variable_topics:
            with col2:
                st.subheader("Selected Variables by Topic")
                st.markdown("Check/uncheck topics and variables to include in your analysis:")

                for topic in st.session_state.variable_topics:
                    topic_name = topic['topic_name']
                    current_topic_selection = st.session_state.topic_selections.get(topic_name, True)

                    # Topic-level checkbox
                    new_topic_selection = st.checkbox(
                        f"**📁 {topic_name}**",
                        value=current_topic_selection,
                        key=f"topic_{topic_name}",
                        help=topic['topic_description']
                    )

                    # If topic selection changed, update all variables in that topic
                    if new_topic_selection != current_topic_selection:
                        st.session_state.topic_selections[topic_name] = new_topic_selection
                        for var in topic['variables']:
                            st.session_state.variable_selections[var['variable_name']] = new_topic_selection
                        # Reset relations when variable selection changes
                        st.session_state.consolidated_relations = []
                        st.session_state.relation_selections = {}
                        st.rerun()

                    # Show variables within topic (indented)
                    if new_topic_selection:  # Only show variables if topic is selected
                        for var in topic['variables']:
                            var_name = var['variable_name']
                            unit_display = f" [{var['unit']}]" if var['unit'] else ""
                            current_var_selection = st.session_state.variable_selections.get(var_name, True)

                            new_var_selection = st.checkbox(
                                f"    📊 {var_name}{unit_display}",
                                value=current_var_selection,
                                key=f"var_{var_name}",
                                help=var['definition']
                            )

                            if new_var_selection != current_var_selection:
                                st.session_state.variable_selections[var_name] = new_var_selection
                                # Reset relations when variable selection changes
                                st.session_state.consolidated_relations = []
                                st.session_state.relation_selections = {}

                                # If variable is unchecked, check if topic should be unchecked
                                if not new_var_selection:
                                    topic_vars_selected = any(
                                        st.session_state.variable_selections.get(v['variable_name'], False)
                                        for v in topic['variables']
                                    )
                                    if not topic_vars_selected:
                                        st.session_state.topic_selections[topic_name] = False

                    st.write("")  # Add spacing between topics

                # Summary statistics
                selected_topics = sum(st.session_state.topic_selections.values())
                selected_vars = sum(st.session_state.variable_selections.values())
                total_topics = len(st.session_state.variable_topics)
                total_vars = sum(len(topic['variables']) for topic in st.session_state.variable_topics)

                st.info(f"Selected: {selected_topics}/{total_topics} topics, {selected_vars}/{total_vars} variables")

    # Step 3: Relation Consolidation
    if st.session_state.variable_topics and any(st.session_state.variable_selections.values()):
        st.header("🔗 Step 3: Relation Consolidation")

        selected_var_names = [
            var_name for var_name, selected in st.session_state.variable_selections.items()
            if selected
        ]

        col1, col2 = st.columns([1, 3])

        with col1:
            if st.button("🔗 Consolidate Relations", key="consolidate_rels"):
                with st.spinner("Calling LLM to consolidate relationships..."):
                    prompt = cld_builder.get_relation_consolidation_prompt(
                        st.session_state.goal_description,
                        selected_var_names,
                        st.session_state.extracted_papers
                    )
                    result, raw_response = cld_builder.call_llm(prompt)

                    if result and 'consolidated_relationships' in result:
                        st.session_state.consolidated_relations = result['consolidated_relationships']
                        # Initialize all as selected
                        st.session_state.relation_selections = {
                            f"{rel['causal_variable']}→{rel['effect_variable']}": True
                            for rel in result['consolidated_relationships']
                        }
                        st.success(f"✅ Found {len(result['consolidated_relationships'])} relationships")
                    else:
                        st.error("❌ Failed to get valid response from LLM")
                        with st.expander("View raw response"):
                            st.text(raw_response)

        # Show consolidated relations with checkboxes
        if st.session_state.consolidated_relations:
            with col2:
                st.subheader("Consolidated Relations")
                st.markdown("Check/uncheck relationships to include in your CLD:")

                # CLD filtering option
                enforce_cld_rules = st.checkbox(
                    "🔄 Enforce CLD rules (variables must have ≥2 incoming OR ≥2 outgoing connections)",
                    value=False,
                    key="enforce_cld_rules",
                    help="In proper CLDs, variables should have multiple connections to show system dynamics"
                )

                rel_keys = []
                for rel in st.session_state.consolidated_relations:
                    rel_key = f"{rel['causal_variable']}→{rel['effect_variable']}"

                    # Check if rel_key already exists to avoid duplicates
                    if rel_key in rel_keys:
                        # Warn
                        st.warning(f"Duplicate relationship found: {rel_key}. Skipping.")
                        continue

                    current_selection = st.session_state.relation_selections.get(rel_key, True)

                    # Format polarity indicator
                    polarity_icon = "➕" if rel['polarity'] == 'positive' else "➖"

                    new_selection = st.checkbox(
                        f"{polarity_icon} **{rel['causal_variable']}** → **{rel['effect_variable']}**",
                        value=current_selection,
                        key=f"rel_{rel_key}",
                        help=f"Relationship: {rel['relationship_name']}\nSources: {rel['supporting_citations']}"
                    )

                    st.session_state.relation_selections[rel_key] = new_selection

                selected_rel_count = sum(st.session_state.relation_selections.values())
                st.info(f"Selected: {selected_rel_count}/{len(st.session_state.consolidated_relations)} relationships")

    # Step 4: CLD Generation
    if st.session_state.consolidated_relations and any(st.session_state.relation_selections.values()):
        st.header("📊 Step 4: Generate Causal Loop Diagram")

        selected_relations = [
            rel for rel in st.session_state.consolidated_relations
            if st.session_state.relation_selections.get(f"{rel['causal_variable']}→{rel['effect_variable']}", False)
        ]

        # Apply CLD filtering if enabled
        if st.session_state.get('enforce_cld_rules', False):
            # Count connections for each variable
            var_connections = {}
            for rel in selected_relations:
                causal_var = rel['causal_variable']
                effect_var = rel['effect_variable']

                if causal_var not in var_connections:
                    var_connections[causal_var] = {'incoming': 0, 'outgoing': 0}
                if effect_var not in var_connections:
                    var_connections[effect_var] = {'incoming': 0, 'outgoing': 0}

                var_connections[causal_var]['outgoing'] += 1
                var_connections[effect_var]['incoming'] += 1

            # Filter variables that have at least 2 incoming OR 2 outgoing connections
            valid_vars = set()
            for var, connections in var_connections.items():
                if connections['incoming'] >= 2 or connections['outgoing'] >= 2:
                    valid_vars.add(var)

            # Filter relations to only include those between valid variables
            original_count = len(selected_relations)
            selected_relations = [
                rel for rel in selected_relations
                if rel['causal_variable'] in valid_vars and rel['effect_variable'] in valid_vars
            ]

            if original_count != len(selected_relations):
                st.info(f"🔄 CLD filtering: Reduced from {original_count} to {len(selected_relations)} relationships")
                excluded_vars = set()
                for rel in st.session_state.consolidated_relations:
                    if st.session_state.relation_selections.get(f"{rel['causal_variable']}→{rel['effect_variable']}",
                                                                False):
                        if rel['causal_variable'] not in valid_vars:
                            excluded_vars.add(rel['causal_variable'])
                        if rel['effect_variable'] not in valid_vars:
                            excluded_vars.add(rel['effect_variable'])

                if excluded_vars:
                    with st.expander("View excluded variables"):
                        for var in sorted(excluded_vars):
                            connections = var_connections.get(var, {'incoming': 0, 'outgoing': 0})
                            st.write(f"• {var} (in: {connections['incoming']}, out: {connections['outgoing']})")

        col1, col2 = st.columns([1, 2])

        with col1:
            layout_type = st.selectbox(
                "Layout Algorithm:",
                options=['Kamada-Kawai', 'Spring', 'Circular', 'Shell', 'Spectral'],
                index=0,
                help="Spring: Force-directed layout, Circular: Nodes in circle, Shell: Concentric shells, Kamada-Kawai: Optimized for aesthetics, Spectral: Based on eigenvectors."
            )

            if st.button("📊 Generate CLD", key="generate_cld"):
                with st.spinner("Generating Causal Loop Diagram..."):
                    fig = cld_builder.create_cld_diagram(selected_relations, layout_type)

                    if fig:
                        st.session_state.cld_figure = fig
                        st.success("✅ CLD generated successfully!")
                    else:
                        st.error("❌ Failed to generate CLD")

        with col2:
            if hasattr(st.session_state, 'cld_figure'):
                st.subheader("Statistics")
                st.write(
                    f"Variables: {len(set([r['causal_variable'] for r in selected_relations] + [r['effect_variable'] for r in selected_relations]))}")
                st.write(f"Relationships: {len(selected_relations)}")
                positive_count = sum(1 for r in selected_relations if r['polarity'] == 'positive')
                negative_count = len(selected_relations) - positive_count
                st.write(f"Positive: {positive_count}, Negative: {negative_count}")

        # Display CLD
        if hasattr(st.session_state, 'cld_figure'):
            st.subheader("Causal Loop Diagram")
            st.plotly_chart(st.session_state.cld_figure, use_container_width=True)

            # Feedback loops analysis
            feedback_loops = cld_builder.detect_feedback_loops(selected_relations)

            if feedback_loops:
                st.subheader("🔄 Feedback Loops")

                for loop in feedback_loops[:10]:  # Show first 10
                    loop_type_icon = "🔄" if loop['loop_type'] == 'reinforcing' else "⚖️"
                    st.write(f"{loop_type_icon} **{loop['loop_type'].title()} Loop** (Length: {loop['length']})")
                    st.write(f"Path: {loop['path']}")
                    st.write("---")

                if len(feedback_loops) > 10:
                    st.info(f"... and {len(feedback_loops) - 10} more loops")

            # Download buttons
            st.subheader("📥 Downloads")

            col1, col2 = st.columns(2)

            with col1:
                # Download diagram as HTML
                html_buffer = io.StringIO()
                st.session_state.cld_figure.write_html(html_buffer)
                html_data = html_buffer.getvalue()

                st.download_button(
                    label="📊 Download Interactive CLD (HTML)",
                    data=html_data,
                    file_name="causal_loop_diagram.html",
                    mime="text/html"
                )

            with col2:
                # Download data as JSON
                export_data = {
                    'goal_description': st.session_state.goal_description,
                    'selected_variables': [var for var in st.session_state.selected_variables
                                           if st.session_state.variable_selections.get(var['variable_name'], False)],
                    'consolidated_relationships': selected_relations,
                    'feedback_loops': feedback_loops
                }

                json_data = json.dumps(export_data, indent=2, ensure_ascii=False)

                st.download_button(
                    label="📄 Download CLD Data (JSON)",
                    data=json_data,
                    file_name="cld_data.json",
                    mime="application/json"
                )


if __name__ == "__main__":
    main()
