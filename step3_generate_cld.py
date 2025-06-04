# Step 3: Generate Causal Loop Diagrams from JSON
# pip install networkx plotly pandas matplotlib kaleido

import os
import json
import networkx as nx
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import re
from collections import Counter


class CLDGenerator:
    """Class to generate Causal Loop Diagrams from consolidated JSON relations."""

    def __init__(self, output_dir="results"):
        self.output_dir = output_dir
        self.graph = nx.DiGraph()
        self.consolidated_data = None
        self.variables_info = {}
        self.pos = None

        # Color schemes
        self.colors = {
            'positive': '#2E8B57',  # Sea Green
            'negative': '#DC143C',  # Crimson
            'node': '#4A90E2',  # Blue
            'node_border': '#2C5AA0',  # Darker Blue
            'reinforcing_loop': '#FF6B35',  # Orange Red
            'balancing_loop': '#6B73FF'  # Blue Violet
        }

    def load_consolidated_data(self, json_path):
        """Load consolidated data from Step 2 JSON."""
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                self.consolidated_data = json.load(f)

            # Build variables info dictionary
            for var in self.consolidated_data.get('selected_variables', []):
                self.variables_info[var['variable_name']] = var['definition']

            num_vars = len(self.consolidated_data.get('selected_variables', []))
            num_rels = len(self.consolidated_data.get('consolidated_relationships', []))
            print(f"Loaded {num_vars} variables and {num_rels} relationships")
            return True

        except Exception as e:
            print(f"Error loading consolidated data: {e}")
            return False

    def build_graph(self):
        """Build NetworkX graph from consolidated relationships."""
        if not self.consolidated_data:
            print("No consolidated data loaded")
            return False

        print("Building causal graph...")

        # Add nodes with variable information
        for var in self.consolidated_data.get('selected_variables', []):
            self.graph.add_node(
                var['variable_name'],
                definition=var['definition']
            )

        # Add edges from relationships
        for rel in self.consolidated_data.get('consolidated_relationships', []):
            causal_var = rel['causal_variable']
            effect_var = rel['effect_variable']

            # Count supporting papers
            citations = rel['supporting_citations']
            num_papers = len(re.findall(r'\([^)]+\)', citations))

            self.graph.add_edge(
                causal_var,
                effect_var,
                polarity=rel['polarity'],
                relationship_name=rel['relationship_name'],
                citations=citations,
                weight=num_papers,
                num_papers=num_papers
            )

        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        return True

    def detect_feedback_loops(self):
        """Detect and analyze feedback loops."""
        print("Detecting feedback loops...")

        try:
            cycles = list(nx.simple_cycles(self.graph))

            cycle_analysis = []
            for i, cycle in enumerate(cycles):
                if len(cycle) <= 10:  # Reasonable cycle length
                    cycle_edges = []
                    cycle_polarities = []
                    cycle_citations = []

                    # Analyze cycle edges
                    for j in range(len(cycle)):
                        start = cycle[j]
                        end = cycle[(j + 1) % len(cycle)]
                        if self.graph.has_edge(start, end):
                            edge_data = self.graph[start][end]
                            cycle_edges.append((start, end))
                            cycle_polarities.append(edge_data['polarity'])
                            cycle_citations.extend(re.findall(r'\([^)]+\)', edge_data['citations']))

                    # Determine loop type
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

            # Sort by length (shorter loops first)
            cycle_analysis.sort(key=lambda x: x['length'])

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

        print(f"Creating interactive Plotly diagram with {layout_type} layout...")

        # Calculate layout
        if layout_type == 'spring':
            self.pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
        elif layout_type == 'circular':
            self.pos = nx.circular_layout(self.graph)
        elif layout_type == 'shell':
            self.pos = nx.shell_layout(self.graph)
        else:
            self.pos = nx.spring_layout(self.graph, seed=42)

        # Prepare node traces
        node_x, node_y, node_text, node_info, node_sizes = [], [], [], [], []

        for node in self.graph.nodes():
            x, y = self.pos[node]
            node_x.append(x)
            node_y.append(y)

            # Node text (shortened for display)
            display_text = node if len(node) <= 20 else node[:17] + "..."
            node_text.append(display_text)

            # Node info for hover
            definition = self.graph.nodes[node].get('definition', 'No definition available')
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)

            hover_text = f"<b>{node}</b><br>"
            hover_text += f"Definition: {definition}<br>"
            hover_text += f"Incoming: {in_degree} | Outgoing: {out_degree}"

            node_info.append(hover_text)

            # Node size based on degree centrality
            degree_centrality = nx.degree_centrality(self.graph)[node]
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

            for edge in self.graph.edges(data=True):
                if edge[2]['polarity'] == polarity:
                    x0, y0 = self.pos[edge[0]]
                    x1, y1 = self.pos[edge[1]]
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])

            if edge_x:  # Only create trace if there are edges of this type
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
                title=dict(
                    text='Causal Loop Diagram',
                    x=0.5,
                    font=dict(size=18)
                ),
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20, l=5, r=5, t=60),
                annotations=[
                    dict(
                        text="Green: Positive relationships | Red: Negative relationships<br>" +
                             "Node size reflects degree centrality",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(size=11)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white',
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
        )

        return fig

    def create_matplotlib_diagram(self, layout_type='spring', figsize=(16, 12)):
        """Create publication-ready matplotlib visualization."""
        if not self.graph.nodes():
            print("No graph to visualize")
            return None

        print(f"Creating static matplotlib diagram with {layout_type} layout...")

        plt.style.use('default')
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')

        # Use existing layout or create new one
        if self.pos is None:
            if layout_type == 'spring':
                self.pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)
            elif layout_type == 'circular':
                self.pos = nx.circular_layout(self.graph)
            elif layout_type == 'shell':
                self.pos = nx.shell_layout(self.graph)

        # Calculate node sizes based on centrality
        degree_centrality = nx.degree_centrality(self.graph)
        node_sizes = [2000 + degree_centrality[node] * 3000 for node in self.graph.nodes()]

        # Draw edges by polarity
        positive_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['polarity'] == 'positive']
        negative_edges = [(u, v) for u, v, d in self.graph.edges(data=True) if d['polarity'] == 'negative']

        # Draw positive edges
        if positive_edges:
            nx.draw_networkx_edges(
                self.graph, self.pos, edgelist=positive_edges,
                edge_color=self.colors['positive'], width=2,
                arrows=True, arrowsize=20, arrowstyle='->', ax=ax
            )

        # Draw negative edges
        if negative_edges:
            nx.draw_networkx_edges(
                self.graph, self.pos, edgelist=negative_edges,
                edge_color=self.colors['negative'], width=2,
                arrows=True, arrowsize=20, arrowstyle='->', ax=ax,
                style='dashed'
            )

        # Draw nodes
        nx.draw_networkx_nodes(
            self.graph, self.pos,
            node_color=self.colors['node'],
            node_size=node_sizes,
            alpha=0.9, ax=ax,
            edgecolors=self.colors['node_border'],
            linewidths=2
        )

        # Draw labels with better formatting
        labels = {}
        for node in self.graph.nodes():
            # Wrap long labels
            if len(node) > 25:
                words = node.split()
                if len(words) > 1:
                    mid = len(words) // 2
                    labels[node] = '\n'.join([' '.join(words[:mid]), ' '.join(words[mid:])])
                else:
                    labels[node] = node[:22] + '...'
            else:
                labels[node] = node

        nx.draw_networkx_labels(
            self.graph, self.pos, labels,
            font_size=8, font_weight='bold',
            font_color='white', ax=ax
        )

        # Create legend
        legend_elements = [
            mpatches.Patch(color=self.colors['positive'], label='Positive relationships'),
            plt.Line2D([0], [0], color=self.colors['negative'], linewidth=2,
                       linestyle='--', label='Negative relationships')
        ]

        ax.legend(handles=legend_elements, loc='upper right', fontsize=12)
        ax.set_title('Causal Loop Diagram', fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()
        return fig

    def save_all_outputs(self, layout_type='spring'):
        """Generate and save all CLD outputs."""
        print("Generating CLD outputs...")

        # Detect feedback loops
        feedback_loops = self.detect_feedback_loops()

        # Create interactive Plotly diagram
        plotly_fig = self.create_plotly_diagram(layout_type)
        plotly_path = None
        if plotly_fig:
            plotly_path = f"{self.output_dir}/step3_interactive_cld.html"
            plotly_fig.write_html(plotly_path)
            print(f"✓ Interactive diagram saved: {plotly_path}")

        # Create static matplotlib diagram
        matplotlib_fig = self.create_matplotlib_diagram(layout_type)
        matplotlib_path = None
        if matplotlib_fig:
            matplotlib_path = f"{self.output_dir}/step3_static_cld.png"
            matplotlib_fig.savefig(matplotlib_path, dpi=300, bbox_inches='tight',
                                   facecolor='white', edgecolor='none')
            plt.close(matplotlib_fig)
            print(f"✓ Static diagram saved: {matplotlib_path}")

        # Save feedback loops analysis
        loops_path = None
        if feedback_loops:
            loops_path = f"{self.output_dir}/step3_feedback_loops.json"
            with open(loops_path, 'w', encoding='utf-8') as f:
                json.dump(feedback_loops, f, indent=2, ensure_ascii=False)
            print(f"✓ Feedback loops analysis saved: {loops_path}")

        # Save basic graph statistics
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
            'interactive_diagram': plotly_path,
            'static_diagram': matplotlib_path,
            'feedback_loops': loops_path,
            'statistics': stats_path
        }


def process_step3(consolidated_json_path, output_dir="results", layout_type='spring'):
    """
    Step 3: Generate Causal Loop Diagrams from consolidated JSON.

    Args:
        consolidated_json_path (str): Path to consolidated relations JSON from Step 2
        output_dir (str): Directory to save results
        layout_type (str): Layout algorithm ('spring', 'circular', 'shell')
    """

    os.makedirs(output_dir, exist_ok=True)

    print(f"Processing consolidated relations from: {consolidated_json_path}")

    # Initialize CLD generator
    cld_generator = CLDGenerator(output_dir)

    # Load consolidated data
    if not cld_generator.load_consolidated_data(consolidated_json_path):
        print("✗ Failed to load consolidated data")
        return

    # Build the causal graph
    if not cld_generator.build_graph():
        print("✗ Failed to build causal graph")
        return

    # Generate all outputs
    output_files = cld_generator.save_all_outputs(layout_type)

    print(f"\n✓ Step 3 completed successfully!")
    print(f"  - Variables: {cld_generator.graph.number_of_nodes()}")
    print(f"  - Relationships: {cld_generator.graph.number_of_edges()}")
    print("  - Generated files:")
    for output_type, filepath in output_files.items():
        if filepath:
            print(f"    • {output_type}: {filepath}")


if __name__ == "__main__":
    # Configuration
    consolidated_json_path = "results/step2_consolidated_relations.json"

    # Run Step 3
    process_step3(
        consolidated_json_path=consolidated_json_path,
        output_dir="results",
        layout_type='spring'  # Options: 'spring', 'circular', 'shell'
    )
