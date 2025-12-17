#!/usr/bin/env python3
"""
Generate a 2D visualization of embeddings for presentation slides.
Shows how similar concepts cluster together in vector space.

Usage:
    python visualize_embeddings.py

Output:
    - embeddings_visualization.png (light theme)
    - embeddings_visualization_dark.png (dark theme)
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from sentence_transformers import SentenceTransformer

# Sample texts to visualize
texts = {
    # Kubernetes deployment concepts
    "deployment": "Kubernetes deployment",
    "replicaset": "ReplicaSet controller",
    "scaling": "horizontal pod autoscaling",
    "replicas": "replicas: 3",
    "rollout": "deployment rollout strategy",

    # Storage concepts
    "pvc": "PersistentVolumeClaim",
    "persistent volume": "persistent volume storage",
    "storage class": "StorageClass provisioner",

    # Service concepts
    "service": "Kubernetes Service",
    "ingress": "Ingress controller",
    "loadbalancer": "LoadBalancer type",

    # Unrelated concepts
    "coffee": "morning coffee break",
    "weather": "sunny weather today",
    "music": "listening to music",
}

# Color mapping by category
colors = {
    "deployment": "#4CAF50", "replicaset": "#4CAF50", "scaling": "#4CAF50",
    "replicas": "#4CAF50", "rollout": "#4CAF50",
    "pvc": "#FF9800", "persistent volume": "#FF9800", "storage class": "#FF9800",
    "service": "#2196F3", "ingress": "#2196F3", "loadbalancer": "#2196F3",
    "coffee": "#F44336", "weather": "#F44336", "music": "#F44336",
}

def generate_embedding_plot():
    print("Loading embedding model...")
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    print("Generating embeddings...")
    labels = list(texts.keys())
    sentences = list(texts.values())
    embeddings = model.encode(sentences)

    print(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}")

    print("Reducing dimensions with t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=5)
    embeddings_2d = tsne.fit_transform(embeddings)

    # ==================== LIGHT THEME ====================
    plt.figure(figsize=(14, 10))

    # Plot each point with its category color
    for i, label in enumerate(labels):
        plt.scatter(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            c=colors[label],
            s=300,
            alpha=0.7,
            edgecolors='white',
            linewidth=2,
            zorder=2
        )
        plt.annotate(
            label,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(12, 8),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold',
            color='#333333'
        )

    # Draw circles around clusters
    from matplotlib.patches import Circle
    ax = plt.gca()

    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=15, label='Deployment/Scaling'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', markersize=15, label='Storage'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=15, label='Networking'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=15, label='Unrelated'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

    # Clean axes - no title, no insight box
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)

    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save light theme
    output_file = 'embeddings_visualization.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"\nSaved: {output_file}")

    # ==================== DARK THEME ====================
    plt.figure(figsize=(14, 10))
    plt.style.use('dark_background')

    for i, label in enumerate(labels):
        plt.scatter(
            embeddings_2d[i, 0],
            embeddings_2d[i, 1],
            c=colors[label],
            s=300,
            alpha=0.85,
            edgecolors='white',
            linewidth=2,
            zorder=2
        )
        plt.annotate(
            label,
            (embeddings_2d[i, 0], embeddings_2d[i, 1]),
            xytext=(12, 8),
            textcoords='offset points',
            fontsize=12,
            fontweight='bold',
            color='white'
        )

    ax = plt.gca()

    # Legend for dark theme
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#4CAF50', markersize=15, label='Deployment/Scaling'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#FF9800', markersize=15, label='Storage'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=15, label='Networking'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=15, label='Unrelated'),
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.8)

    # Clean axes - no title, no insight box
    plt.xlabel('Dimension 1', fontsize=12, color='white')
    plt.ylabel('Dimension 2', fontsize=12, color='white')

    plt.grid(True, alpha=0.2, color='gray')
    plt.tight_layout()

    output_file_dark = 'embeddings_visualization_dark.png'
    plt.savefig(output_file_dark, dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
    print(f"Saved: {output_file_dark}")

    print("\n✅ Done! Use these images in your slides.")
    print("   - Light theme: embeddings_visualization.png")
    print("   - Dark theme:  embeddings_visualization_dark.png")

if __name__ == "__main__":
    generate_embedding_plot()
