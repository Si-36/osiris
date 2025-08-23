#!/usr/bin/env python3
"""
📊 AURA INTELLIGENCE SYSTEM VISUALIZATION
========================================

Visualize the working components and data flow.
"""

import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.patches import Rectangle
import numpy as np

def create_system_diagram():
    """Create system architecture diagram"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # ============================================================================
    # COMPONENT STATUS CHART
    # ============================================================================
    
    # Data from our test
    categories = ['Core', 'Neural', 'Consciousness', 'Agents', 'Memory', 'TDA', 
                 'Orchestration', 'Communication', 'Observability', 'Resilience', 'API']
    working = [2, 3, 3, 3, 3, 2, 2, 0, 3, 1, 2]  # Working components per category
    total = [3, 4, 3, 4, 4, 3, 3, 2, 3, 3, 2]    # Total components per category
    
    x = np.arange(len(categories))
    width = 0.35
    
    ax1.bar(x - width/2, working, width, label='Working', color='green', alpha=0.7)
    ax1.bar(x + width/2, [t-w for t, w in zip(total, working)], width, 
            label='Needs Fix', color='red', alpha=0.7, bottom=working)
    
    ax1.set_xlabel('Component Categories')
    ax1.set_ylabel('Number of Components')
    ax1.set_title('AURA Intelligence System Status\n70.6% Working (24/34 components)')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add success rate text
    ax1.text(0.02, 0.98, 'Success Rate: 70.6%\nPipeline: ✅ Working', 
             transform=ax1.transAxes, fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # ============================================================================
    # DATA FLOW DIAGRAM
    # ============================================================================
    
    # Create network graph
    G = nx.DiGraph()
    
    # Add nodes with positions
    nodes = {
        'API': (0, 2),
        'Unified System': (1, 2),
        'Consciousness': (2, 3),
        'Memory': (2, 1),
        'Neural': (3, 3),
        'Agents': (3, 1),
        'TDA': (4, 2),
        'Response': (5, 2)
    }
    
    # Add nodes
    for node, pos in nodes.items():
        G.add_node(node, pos=pos)
    
    # Add edges (data flow)
    edges = [
        ('API', 'Unified System'),
        ('Unified System', 'Consciousness'),
        ('Unified System', 'Memory'),
        ('Consciousness', 'Neural'),
        ('Memory', 'Agents'),
        ('Neural', 'TDA'),
        ('Agents', 'TDA'),
        ('TDA', 'Response')
    ]
    
    G.add_edges_from(edges)
    
    # Get positions
    pos = nx.get_node_attributes(G, 'pos')
    
    # Draw the graph
    nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightblue', 
            node_size=2000, font_size=10, font_weight='bold',
            edge_color='gray', arrows=True, arrowsize=20)
    
    ax2.set_title('AURA Intelligence Data Flow\nWorking Pipeline')
    ax2.set_xlim(-0.5, 5.5)
    ax2.set_ylim(0.5, 3.5)
    
    # Add working status indicators
    working_components = ['API', 'Unified System', 'Consciousness', 'Memory']
    for node in working_components:
        x, y = pos[node]
        ax2.text(x, y-0.3, '✅', fontsize=16, ha='center', va='center')
    
    plt.tight_layout()
    plt.savefig('aura_system_status.png', dpi=300, bbox_inches='tight')
    print("  ✅ System diagram saved: aura_system_status.png")
    
    return fig

if __name__ == "__main__":
    create_system_diagram()
    plt.show()
