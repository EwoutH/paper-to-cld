# Step 3: Generate Causal Loop Diagrams
# pip install networkx plotly pandas matplotlib seaborn kaleido

import os
import pandas as pd
import networkx as nx
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from collections import Counter, defaultdict
import json
import re
from pathlib import Path


class CLDGenerator:
    """Class to generate Causal Loop Diagrams from consolidated relations."""

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.graph = nx.DiGraph()
        self.relations_df = None
        self.pos = None

        # Color schemes
        self.colors = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'node': '#4A90E2',  # Blue
            'node_border': '#2C5AA0',
            'background': '#FAFAFA',
            'reinforcing_loop': '#FF6B35',  # Orange
            'balancing_loop': '#6B73FF'  # Purple
        }

    def load_consolidated_relations(self, csv_path):
        """Load consolidated relations from Step 2."""
        try:
            self.relations_df = pd.read_csv(csv_path)
            print(f"Loaded {len(self.relations_df)} consolidated relations")
            return True
        except Exception as e:
            print(f"Error loading relations: {e}")
            return False

    def build_graph(self):
        """Build NetworkX graph from relations."""
        if self.relations_df is None:
            print("No relations loaded")
            return False

        print("Building causal graph...")

        # Add all relationships as edges
        for _, row in self.relations_df.iterrows():
            causal_var = str(row['causal_variable']).strip()
            effect_var = str(row['effect_variable']).strip()
            polarity = str(row['polarity']).strip().lower()
            relationship = str(row['relationship_name']).strip()
            citations = str(row['supporting_citations']).strip()

            # Count supporting papers
            num_papers = len(re.findall(r'\([^)]+\)', citations))

            # Add edge with attributes
            self.graph.add_edge(
                causal_var,
                effect_var,
                polarity=polarity,
                relationship=relationship,
                citations=citations,
                weight=num_papers,
                num_papers=num_papers
            )

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return True

    def detect_feedback_loops(self):
        """Detect and analyze feedback loops in the graph."""
        print("Detecting feedback loops...")

        # Find all simple cycles
        try:
            cycles = list(nx.simple_cycles(self.graph))

            # Analyze cycles
            cycle_analysis = []
            for i, cycle in enumerate(cycles):
                if len(cycle) <= 8:  # Limit to reasonable cycle lengths
                    cycle_edges = []
                    cycle_polarities = []
                    cycle_citations = []

                    # Get edges in the cycle
                    for j in range(len(cycle)):
                        start = cycle[j]
                        end = cycle[(j + 1) % len(cycle)]
                        if self.graph.has_edge(start, end):
                            edge_data = self.graph[start][end]
                            cycle_edges.append((start, end))
                            cycle_polarities.append(edge_data['polarity'])
                            cycle_citations.extend(re.findall(r'\([^)]+\)', edge_data['citations']))

                    # Determine overall loop polarity
                    neg_count = cycle_polarities.count('negative')
                    loop_type = 'reinforcing' if neg_count % 2 == 0 else 'balancing'

                    cycle_analysis.append({
                        'cycle_id': i + 1,
                        'nodes': cycle,
                        'length': len(cycle),
                        'edges': cycle_edges,
                        'polarities': cycle_polarities,
                        'loop_type': loop_type,
                        'negative_links': neg_count,
                        'supporting_citations': list(set(cycle_citations))
                    })

            print(f"Found {len(cycles)} feedback loops")
            return cycle_analysis

        except Exception as e:
            print(f"Error detecting cycles: {e}")
            return []

    def create_plotly_diagram(self, layout_type='spring'):
        """Create interactive Plotly visualization."""
        if not self.graph.nodes():
            print("No graph to visualize")
            return None

        print(f"Creating Plotly diagram with {layout_type} layout...")

        # Calculate layout
        if layout_type == 'spring':
            self.pos = nx.spring_layout(self.graph, k=3, iterations=50)
        elif layout_type == 'circular':
            self.pos = nx.circular_layout(self.graph)
        elif layout_type == 'hierarchical':
            self.pos = nx.shell_layout(self.graph)
        else:
            self.pos = nx.spring_layout(self.graph)

        # Prepare node traces
        node_x = []
        node_y = []
        node_text = []
        node_info = []

        for node in self.graph.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)

            # Count incoming and outgoing edges
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)

            node_info.append(f"{node}<br>Incoming: {in_degree}<br>Outgoing: {out_degree}")

        # Create node trace
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="middle center",
            hoverinfo='text',
            hovertext=node_info,
            marker=dict(
                size=20,
                color=self.colors['node'],
                line=dict(width=2, color=self.colors['node_border'])
            ),
            textfont=dict(size=10, color='white'),
            name='Variables'
        )

        # Prepare edge traces (separate for positive and negative)
        edge_traces = []

        for polarity, color in [('positive', self.colors['positive']), ('negative', self.colors['negative'])]:
            edge_x = []
            edge_y = []
            edge_info = []

            for edge in self.graph.edges(data=True):
                if edge[2]['polarity'] == polarity:
                    x0, y0 = self.pos[edge[0]]
                    x1, y1 = self.pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

                    # Add edge info
                    relationship = edge[2]['relationship']
                    citations = edge[2]['citations']
                    edge_info.append(f"{edge[0]} → {edge[1]}<br>{relationship}<br>{citations}")

            if edge_x:  # Only create trace if there are edges of this type
                edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2, color=color),
                    hoverinfo='none',
                    mode='lines',
                    name=f'{polarity.capitalize()} relationships'
                )
                edge_traces.append(edge_trace)

        # Create figure
        fig = go.Figure(data=edge_traces + [node_trace],
                        layout=go.Layout(
                            title='Causal Loop Diagram',
                            titlefont_size=16,
                            showlegend=True,
                            hovermode='closest',
                            margin=dict(b=20, l=5, r=5, t=40),
                            annotations=[dict(
                                text="Green: Positive relationships, Red: Negative relationships",
                                showarrow=False,
                                xref="paper", yref="paper",
                                x=0.005, y=-0.002,
                                xanchor='left', yanchor='bottom',
                                font=dict(size=12)
                            )],
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            plot_bgcolor='white'
                        ))

        return fig

    def create_matplotlib_diagram(self, layout_type='spring', figsize=(16, 12)):
        """Create publication-ready matplotlib visualization."""
        if not self.graph.nodes():
            print("No graph to visualize")
            return None

        print(f"Creating matplotlib diagram with {layout_type} layout...")

        # Set style
        plt.style.use('default')
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        # Calculate layout if not already done
        if self.pos is None:
            if layout_type == 'spring':
                self.pos = nx.spring_layout(self.graph, k=3, iterations=50)
            elif layout_type == 'circular':
                self.pos = nx.circular_layout(self.graph)
            elif layout_type == 'hierarchical':
                self.pos = nx.shell_layout(self.graph)

        # Draw edges by polarity
        positive_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['polarity'] == 'positive']
        negative_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['polarity'] == 'negative']

        # Draw positive edges
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=positive_edges,
                               edge_color=self.colors['positive'], width=2,
                               arrows=True, arrowsize=20, arrowstyle='->', ax=ax)

        # Draw negative edges
        nx.draw_networkx_edges(self.graph, self.pos, edgelist=negative_edges,
                               edge_color=self.colors['negative'], width=2,
                               arrows=True, arrowsize=20, arrowstyle='->', ax=ax,
                               style='dashed')

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, self.pos, node_color=self.colors['node'],
                               node_size=3000, alpha=0.9, ax=ax)

        # Draw labels
        nx.draw_networkx_labels(self.graph, self.pos, font_size=8, font_weight='bold',
                                font_color='white', ax=ax)

        # Create legend
        positive_patch = mpatches.Patch(color=self.colors['positive'], label='Positive relationship')
        negative_patch = mpatches.Patch(color=self.colors['negative'], label='Negative relationship')
        ax.legend(handles=[positive_patch, negative_patch], loc='upper right')

        ax.set_title('Causal Loop Diagram', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()
        return fig

    def generate_narrative(self, feedback_loops):
        """Generate narrative explanation of the causal system."""
        print("Generating system narrative...")

        narrative = []
        narrative.append("CAUSAL SYSTEM ANALYSIS NARRATIVE")
        narrative.append("=" * 50)
        narrative.append("")

        # System overview
        narrative.append("SYSTEM OVERVIEW:")
        narrative.append(
            f"The causal system contains {self.graph.number_of_nodes()} variables connected by {self.graph.number_of_edges()} causal relationships.")

        # Relationship distribution
        positive_count = sum(1 for _, _, d in self.graph.edges(data=True) if d['polarity'] == 'positive')
        negative_count = sum(1 for _, _, d in self.graph.edges(data=True) if d['polarity'] == 'negative')
        narrative.append(f"Of these relationships, {positive_count} are positive and {negative_count} are negative.")
        narrative.append("")

        # Most influential variables
        narrative.append("MOST INFLUENTIAL VARIABLES:")

        # Calculate centrality measures
        in_centrality = nx.in_degree_centrality(self.graph)
        out_centrality = nx.out_degree_centrality(self.graph)
        betweenness = nx.betweenness_centrality(self.graph)

        # Top variables by different measures
        top_inputs = sorted(out_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_outputs = sorted(in_centrality.items(), key=lambda x: x[1], reverse=True)[:5]
        top_brokers = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

        narrative.append("Key causal drivers (high outgoing connections):")
        for var, centrality in top_inputs:
            out_degree = self.graph.out_degree(var)
            narrative.append(f"  - {var}: influences {out_degree} other variables")
        narrative.append("")

        narrative.append("Key outcome variables (high incoming connections):")
        for var, centrality in top_outputs:
            in_degree = self.graph.in_degree(var)
            narrative.append(f"  - {var}: influenced by {in_degree} other variables")
        narrative.append("")

        narrative.append("Key mediating variables (high betweenness centrality):")
        for var, centrality in top_brokers:
            if centrality > 0:
                narrative.append(f"  - {var}: serves as important pathway between variables")
        narrative.append("")

        # Feedback loops analysis
        narrative.append("FEEDBACK LOOP ANALYSIS:")

        if feedback_loops:
            reinforcing_loops = [loop for loop in feedback_loops if loop['loop_type'] == 'reinforcing']
            balancing_loops = [loop for loop in feedback_loops if loop['loop_type'] == 'balancing']

            narrative.append(
                f"Found {len(reinforcing_loops)} reinforcing loops and {len(balancing_loops)} balancing loops.")
            narrative.append("")

            if reinforcing_loops:
                narrative.append("REINFORCING LOOPS (amplify changes):")
                for loop in reinforcing_loops[:5]:  # Top 5 loops
                    loop_description = " → ".join(loop['nodes']) + f" → {loop['nodes'][0]}"
                    narrative.append(f"  Loop {loop['cycle_id']}: {loop_description}")
                    narrative.append(f"    Length: {loop['length']} variables")
                    if loop['supporting_citations']:
                        narrative.append(f"    Supported by: {', '.join(loop['supporting_citations'][:3])}")
                    narrative.append("")

            if balancing_loops:
                narrative.append("BALANCING LOOPS (stabilize system):")
                for loop in balancing_loops[:5]:  # Top 5 loops
                    loop_description = " → ".join(loop['nodes']) + f" → {loop['nodes'][0]}"
                    narrative.append(f"  Loop {loop['cycle_id']}: {loop_description}")
                    narrative.append(f"    Length: {loop['length']} variables")
                    if loop['supporting_citations']:
                        narrative.append(f"    Supported by: {', '.join(loop['supporting_citations'][:3])}")
                    narrative.append("")
        else:
            narrative.append("No feedback loops detected in the current system.")
            narrative.append("")

        # System behavior implications
        narrative.append("SYSTEM BEHAVIOR IMPLICATIONS:")

        if feedback_loops:
            reinforcing_count = len([loop for loop in feedback_loops if loop['loop_type'] == 'reinforcing'])
            balancing_count = len([loop for loop in feedback_loops if loop['loop_type'] == 'balancing'])

            if reinforcing_count > balancing_count:
                narrative.append("The system is dominated by reinforcing loops, suggesting potential for:")
                narrative.append("  - Exponential growth or decline")
                narrative.append("  - Instability and rapid changes")
                narrative.append("  - Virtuous or vicious cycles")
                narrative.append("  - Need for external intervention to prevent runaway effects")
            elif balancing_count > reinforcing_count:
                narrative.append("The system is dominated by balancing loops, suggesting:")
                narrative.append("  - Self-regulating behavior")
                narrative.append("  - Resistance to change")
                narrative.append("  - Goal-seeking or stability-maintaining tendencies")
                narrative.append("  - Gradual adjustment toward equilibrium")
            else:
                narrative.append("The system has balanced reinforcing and balancing loops, suggesting:")
                narrative.append("  - Complex dynamic behavior")
                narrative.append("  - Potential for both stability and change")
                narrative.append("  - Context-dependent system responses")
        else:
            narrative.append("Without feedback loops, the system exhibits:")
            narrative.append("  - Linear cause-and-effect relationships")
            narrative.append("  - Predictable responses to interventions")
            narrative.append("  - No self-reinforcing dynamics")

        narrative.append("")

        # Policy implications
        narrative.append("POLICY AND INTERVENTION IMPLICATIONS:")

        # Identify leverage points
        high_influence_vars = [var for var, centrality in
                               sorted(out_centrality.items(), key=lambda x: x[1], reverse=True)[:3]]

        narrative.append("High-leverage intervention points:")
        for var in high_influence_vars:
            out_degree = self.graph.out_degree(var)
            narrative.append(f"  - {var}: Changes here could affect {out_degree} other variables")
        narrative.append("")

        if feedback_loops:
            narrative.append("Loop intervention strategies:")
            narrative.append("  - Breaking reinforcing loops at their weakest links")
            narrative.append("  - Strengthening balancing loops to improve system stability")
            narrative.append("  - Monitoring key variables within important feedback loops")

        return "\n".join(narrative)

    def save_all_outputs(self, layout_type='spring'):
        """Generate and save all CLD outputs."""
        print("Generating all CLD outputs...")

        # Detect feedback loops
        feedback_loops = self.detect_feedback_loops()

        # Create Plotly interactive diagram
        plotly_fig = self.create_plotly_diagram(layout_type)
        if plotly_fig:
            plotly_path = f"{self.output_dir}/step3_interactive_cld.html"
            plotly_fig.write_html(plotly_path)
            print(f"✓ Interactive diagram saved: {plotly_path}")

        # Create matplotlib static diagram
        matplotlib_fig = self.create_matplotlib_diagram(layout_type)
        if matplotlib_fig:
            matplotlib_path = f"{self.output_dir}/step3_static_cld.png"
            matplotlib_fig.savefig(matplotlib_path, dpi=300, bbox_inches='tight')
            plt.close(matplotlib_fig)
            print(f"✓ Static diagram saved: {matplotlib_path}")

        # Generate and save narrative
        narrative = self.generate_narrative(feedback_loops)
        narrative_path = f"{self.output_dir}/step3_system_narrative.txt"
        with open(narrative_path, 'w', encoding='utf-8') as f:
            f.write(narrative)
        print(f"✓ System narrative saved: {narrative_path}")

        # Save feedback loops analysis
        if feedback_loops:
            loops_path = f"{self.output_dir}/step3_feedback_loops.json"
            with open(loops_path, 'w', encoding='utf-8') as f:
                json.dump(feedback_loops, f, indent=2)
            print(f"✓ Feedback loops analysis saved: {loops_path}")

        # Save graph statistics
        stats = {
            'nodes': self.graph.number_of_nodes(),
            'edges': self.graph.number_of_edges(),
            'positive_relationships': sum(1 for _, _, d in self.graph.edges(data=True) if d['polarity'] == 'positive'),
            'negative_relationships': sum(1 for _, _, d in self.graph.edges(data=True) if d['polarity'] == 'negative'),
            'feedback_loops': len(feedback_loops),
            'reinforcing_loops': len([loop for loop in feedback_loops if loop['loop_type'] == 'reinforcing']),
            'balancing_loops': len([loop for loop in feedback_loops if loop['loop_type'] == 'balancing']),
            'average_papers_per_relationship': sum(d['num_papers'] for _, _, d in self.graph.edges(
                data=True)) / self.graph.number_of_edges() if self.graph.number_of_edges() > 0 else 0
        }

        stats_path = f"{self.output_dir}/step3_graph_statistics.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2)
        print(f"✓ Graph statistics saved: {stats_path}")

        return {
            'plotly_diagram': plotly_path if plotly_fig else None,
            'static_diagram': matplotlib_path if matplotlib_fig else None,
            'narrative': narrative_path,
            'feedback_loops': loops_path if feedback_loops else None,
            'statistics': stats_path
        }


def process_step3(consolidated_csv_path, output_dir="results", layout_type='spring'):
    """
    Step 3: Generate Causal Loop Diagrams from consolidated relations.

    Args:
        consolidated_csv_path (str): Path to consolidated relations CSV from Step 2
        output_dir (str): Directory to save results
        layout_type (str): Layout algorithm ('spring', 'circular', 'hierarchical')
    """

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing consolidated relations from: {consolidated_csv_path}")

    # Initialize CLD generator
    cld_generator = CLDGenerator(output_dir)

    # Load consolidated relations
    if not cld_generator.load_consolidated_relations(consolidated_csv_path):
        print("✗ Failed to load consolidated relations")
        return

    # Build the causal graph
    if not cld_generator.build_graph():
        print("✗ Failed to build causal graph")
        return

    # Generate all outputs
    output_files = cld_generator.save_all_outputs(layout_type)

    print(f"\n✓ Step 3 completed successfully!")
    print(f"  - Graph: {cld_generator.graph.number_of_nodes()} nodes, {cld_generator.graph.number_of_edges()} edges")
    print("  - Generated files:")
    for output_type, filepath in output_files.items():
        if filepath:
            print(f"    - {output_type}: {filepath}")


if __name__ == "__main__":
    # Configuration
    consolidated_csv_path = "results/step2_consolidated_relations.csv"

    # Run Step 3
    process_step3(
        consolidated_csv_path=consolidated_csv_path,
        output_dir="results",
        layout_type='spring'  # Options: 'spring', 'circular', 'hierarchical'
    )
