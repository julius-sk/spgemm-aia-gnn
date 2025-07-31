#!/usr/bin/env python3
"""
Graph Data Generator for SpGEMM-AIA Testing
============================================

This script generates graph adjacency matrices for the specified datasets
in a format that can be easily loaded by the C++ SpGEMM test program.
"""

import numpy as np
import scipy.sparse as sp
import os
import pickle
import argparse
from typing import Tuple, Dict
import networkx as nx

# Dataset configurations matching the C++ test file
DATASETS = {
    'reddit': {
        'nodes': 232965,
        'edges': 11606919,
        'description': 'Reddit dataset from DGL'
    },
    'flickr': {
        'nodes': 89250, 
        'edges': 899756,
        'description': 'Flickr dataset from DGL'
    },
    'yelp': {
        'nodes': 716847,
        'edges': 6977410, 
        'description': 'Yelp dataset from DGL'
    },
    'ogbn-products': {
        'nodes': 2449029,
        'edges': 61859140,
        'description': 'OGBN-Products dataset'
    },
    'ogbn-proteins': {
        'nodes': 132534,
        'edges': 39561252,
        'description': 'OGBN-Proteins dataset'
    }
}

def generate_power_law_graph(n_nodes: int, n_edges: int, alpha: float = 2.5, 
                           seed: int = 42) -> sp.csr_matrix:
    """
    Generate a synthetic graph with power-law degree distribution.
    
    Args:
        n_nodes: Number of nodes
        n_edges: Target number of edges
        alpha: Power-law exponent
        seed: Random seed
        
    Returns:
        Adjacency matrix in CSR format
    """
    np.random.seed(seed)
    
    print(f"Generating power-law graph with {n_nodes} nodes, target {n_edges} edges")
    
    # Generate degree sequence following power-law distribution
    degrees = np.random.zipf(alpha, n_nodes)
    
    # Scale degrees to match target edge count
    total_degree = np.sum(degrees)
    scale_factor = (2 * n_edges) / total_degree
    degrees = (degrees * scale_factor).astype(int)
    
    # Ensure degrees are valid
    degrees = np.clip(degrees, 1, n_nodes - 1)
    
    # Make total degree even for simple graph generation
    if np.sum(degrees) % 2 == 1:
        degrees[0] += 1
    
    print(f"Degree statistics: min={np.min(degrees)}, max={np.max(degrees)}, "
          f"mean={np.mean(degrees):.2f}, total={np.sum(degrees)}")
    
    # Generate graph using configuration model
    try:
        G = nx.configuration_model(degrees, seed=seed)
        G = nx.Graph(G)  # Remove multi-edges and self-loops
        G.remove_edges_from(nx.selfloop_edges(G))
        
        # Convert to adjacency matrix
        adj_matrix = nx.adjacency_matrix(G, nodelist=range(n_nodes))
        
        print(f"Generated graph with {adj_matrix.nnz} edges "
              f"(target was {n_edges}, efficiency: {adj_matrix.nnz/n_edges:.2%})")
        
        return adj_matrix.tocsr()
        
    except Exception as e:
        print(f"Warning: Configuration model failed ({e}), using Erdős-Rényi fallback")
        
        # Fallback to Erdős-Rényi model
        p = n_edges / (n_nodes * (n_nodes - 1) / 2)
        p = min(p, 0.1)  # Cap probability for memory efficiency
        
        G = nx.erdos_renyi_graph(n_nodes, p, seed=seed)
        adj_matrix = nx.adjacency_matrix(G, nodelist=range(n_nodes))
        
        print(f"Fallback: Generated Erdős-Rényi graph with {adj_matrix.nnz} edges")
        
        return adj_matrix.tocsr()

def save_csr_matrix(matrix: sp.csr_matrix, filename: str) -> None:
    """
    Save CSR matrix in binary format that can be easily loaded by C++.
    
    Args:
        matrix: CSR matrix to save
        filename: Output filename
    """
    print(f"Saving matrix to {filename}")
    print(f"Matrix shape: {matrix.shape}, nnz: {matrix.nnz}")
    
    # Save in NPZ format (easy to load in both Python and C++)
    np.savez_compressed(filename,
                       shape=np.array(matrix.shape, dtype=np.int32),
                       indptr=matrix.indptr.astype(np.int32),
                       indices=matrix.indices.astype(np.int32), 
                       data=matrix.data.astype(np.float32))
    
    print(f"✅ Saved matrix: {matrix.shape[0]}x{matrix.shape[1]}, {matrix.nnz} nnz")

def load_csr_matrix(filename: str) -> sp.csr_matrix:
    """
    Load CSR matrix from binary format.
    
    Args:
        filename: Input filename
        
    Returns:
        Loaded CSR matrix
    """
    data = np.load(filename)
    shape = tuple(data['shape'])
    
    matrix = sp.csr_matrix((data['data'], data['indices'], data['indptr']), 
                          shape=shape)
    
    print(f"✅ Loaded matrix: {matrix.shape[0]}x{matrix.shape[1]}, {matrix.nnz} nnz")
    return matrix

def generate_random_sparse_features(n_nodes: int, n_features: int = 256, 
                                  sparsity: float = 0.5, seed: int = 123) -> sp.csr_matrix:
    """
    Generate random sparse feature matrix.
    
    Args:
        n_nodes: Number of nodes
        n_features: Number of features (default 1024)
        sparsity: Sparsity level (fraction of non-zero elements)
        seed: Random seed
        
    Returns:
        Sparse feature matrix in CSR format
    """
    np.random.seed(seed)
    
    n_nonzeros = int(n_nodes * n_features * sparsity)
    
    # Generate random indices
    row_indices = np.random.choice(n_nodes, size=n_nonzeros, replace=True)
    col_indices = np.random.choice(n_features, size=n_nonzeros, replace=True)
    
    # Generate random values
    values = np.random.uniform(0, 1, size=n_nonzeros).astype(np.float32)
    
    # Create sparse matrix
    feature_matrix = sp.coo_matrix((values, (row_indices, col_indices)), 
                                  shape=(n_nodes, n_features))
    
    # Convert to CSR and remove duplicates
    feature_matrix = feature_matrix.tocsr()
    feature_matrix.eliminate_zeros()
    
    actual_sparsity = feature_matrix.nnz / (n_nodes * n_features)
    print(f"Generated feature matrix: {n_nodes}x{n_features}, "
          f"nnz: {feature_matrix.nnz}, sparsity: {actual_sparsity:.4f}")
    
    return feature_matrix

def generate_all_datasets(output_dir: str = "graph_data") -> None:
    """
    Generate all datasets and save them.
    
    Args:
        output_dir: Directory to save generated graphs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for dataset_name, config in DATASETS.items():
        print(f"\n{'='*60}")
        print(f"Generating dataset: {dataset_name}")
        print(f"Description: {config['description']}")
        print(f"{'='*60}")
        
        # Generate adjacency matrix
        adj_matrix = generate_power_law_graph(
            config['nodes'], 
            config['edges'], 
            alpha=2.5,
            seed=42
        )
        
        # Save adjacency matrix
        adj_filename = os.path.join(output_dir, f"{dataset_name}_adj.npz")
        save_csr_matrix(adj_matrix, adj_filename)
        
        # Generate and save feature matrices for different sparsity levels (k/256)
        sparsity_levels = [8/256, 16/256, 32/256, 64/256, 128/256]
        
        for sparsity in sparsity_levels:
            k_value = int(sparsity * 256)
            print(f"\nGenerating features with k={k_value}/256 sparsity ({sparsity:.4f})")
            features = generate_random_sparse_features(
                config['nodes'], 
                256, 
                sparsity, 
                seed=123
            )
            
            feat_filename = os.path.join(output_dir, 
                                       f"{dataset_name}_features_k{k_value}.npz")
            save_csr_matrix(features, feat_filename)
        
        print(f"✅ Completed {dataset_name}")

def verify_datasets(data_dir: str = "graph_data") -> None:
    """
    Verify that all generated datasets can be loaded correctly.
    
    Args:
        data_dir: Directory containing the generated graphs
    """
    print(f"\n{'='*60}")
    print("Verifying generated datasets")
    print(f"{'='*60}")
    
    for dataset_name in DATASETS.keys():
        print(f"\nVerifying {dataset_name}:")
        
        # Check adjacency matrix
        adj_file = os.path.join(data_dir, f"{dataset_name}_adj.npz")
        if os.path.exists(adj_file):
            adj_matrix = load_csr_matrix(adj_file)
            print(f"  Adjacency: ✅ {adj_matrix.shape}, {adj_matrix.nnz} edges")
        else:
            print(f"  Adjacency: ❌ File not found: {adj_file}")
        
        # Check feature matrices
        sparsity_levels = [8/256, 16/256, 32/256, 64/256, 128/256]
        for sparsity in sparsity_levels:
            k_value = int(sparsity * 256)
            feat_file = os.path.join(data_dir, f"{dataset_name}_features_k{k_value}.npz")
            if os.path.exists(feat_file):
                feat_matrix = load_csr_matrix(feat_file)
                actual_sparsity = feat_matrix.nnz / (feat_matrix.shape[0] * feat_matrix.shape[1])
                print(f"  Features k={k_value}/256: ✅ {feat_matrix.shape}, "
                      f"nnz: {feat_matrix.nnz}, actual sparsity: {actual_sparsity:.4f}")
            else:
                print(f"  Features k={k_value}/256: ❌ File not found: {feat_file}")

def main():
    parser = argparse.ArgumentParser(description="Generate graph datasets for SpGEMM testing")
    parser.add_argument("--output-dir", "-o", default="graph_data", 
                       help="Output directory for generated graphs")
    parser.add_argument("--verify-only", "-v", action="store_true",
                       help="Only verify existing datasets")
    parser.add_argument("--dataset", "-d", choices=list(DATASETS.keys()),
                       help="Generate only specific dataset")
    
    args = parser.parse_args()
    
    if args.verify_only:
        verify_datasets(args.output_dir)
    else:
        if args.dataset:
            # Generate single dataset
            dataset_name = args.dataset
            config = DATASETS[dataset_name]
            
            os.makedirs(args.output_dir, exist_ok=True)
            
            print(f"Generating dataset: {dataset_name}")
            adj_matrix = generate_power_law_graph(config['nodes'], config['edges'])
            
            adj_filename = os.path.join(args.output_dir, f"{dataset_name}_adj.npz")
            save_csr_matrix(adj_matrix, adj_filename)
            
            sparsity_levels = [8/256, 16/256, 32/256, 64/256, 128/256]
            for sparsity in sparsity_levels:
                k_value = int(sparsity * 256)
                features = generate_random_sparse_features(config['nodes'], 256, sparsity)
                feat_filename = os.path.join(args.output_dir, 
                                           f"{dataset_name}_features_k{k_value}.npz")
                save_csr_matrix(features, feat_filename)
        else:
            # Generate all datasets
            generate_all_datasets(args.output_dir)
        
        # Verify after generation
        verify_datasets(args.output_dir)
    
    print(f"\n✅ Done! Generated graphs are in '{args.output_dir}' directory")

if __name__ == "__main__":
    main()
