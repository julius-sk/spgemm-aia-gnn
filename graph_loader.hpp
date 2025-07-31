// graph_loader.hpp - Load graphs from .indices and .indptr files
// Based on spgemm-pruning loading format

#ifndef GRAPH_LOADER_HPP
#define GRAPH_LOADER_HPP

#include <string>
#include <vector>
#include <fstream>
#include <iostream>
#include <random>
#include <CSR.hpp>

template<typename T>
std::vector<T> read_binary_array(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file) {
        throw std::runtime_error("Cannot open file: " + filepath);
    }
    
    // Get file size
    file.seekg(0, std::ios::end);
    size_t file_size = file.tellg();
    file.seekg(0, std::ios::beg);
    
    // Calculate number of elements
    size_t num_elements = file_size / sizeof(T);
    
    // Read data
    std::vector<T> data(num_elements);
    file.read(reinterpret_cast<char*>(data.data()), file_size);
    
    return data;
}

template<class idType, class valType>
CSR<idType, valType> load_graph_from_spgemm_pruning(const std::string& data_path, 
                                                    const std::string& graph_name) {
    CSR<idType, valType> graph;
    
    // Construct file paths
    std::string indptr_file = data_path + "/" + graph_name + ".indptr";
    std::string indices_file = data_path + "/" + graph_name + ".indices";
    
    std::cout << "Loading graph: " << graph_name << std::endl;
    std::cout << "  indptr file: " << indptr_file << std::endl;
    std::cout << "  indices file: " << indices_file << std::endl;
    
    try {
        // Read indptr and indices arrays
        std::vector<idType> indptr = read_binary_array<idType>(indptr_file);
        std::vector<idType> indices = read_binary_array<idType>(indices_file);
        
        // Calculate graph statistics
        idType v_num = indptr.size() - 1;  // Number of vertices
        idType e_num = indices.size();     // Number of edges
        
        std::cout << "  Vertices: " << v_num << std::endl;
        std::cout << "  Edges: " << e_num << std::endl;
        std::cout << "  Average degree: " << (float)e_num / v_num << std::endl;
        
        // Generate random values (similar to main.cu input_mode=1)
        std::vector<valType> values(e_num);
        std::mt19937 gen(123);  // Same seed as main.cu
        std::uniform_real_distribution<valType> dist(0.0, 1.0);
        for (idType i = 0; i < e_num; i++) {
            values[i] = dist(gen);
        }
        
        // Initialize CSR structure
        graph.nrow = v_num;
        graph.ncolumn = v_num;  // Use ncolumn (not ncol)
        graph.nnz = e_num;
        graph.host_malloc = true;
        graph.device_malloc = false;
        
        // Allocate CPU memory manually (no alloc_cpu_csr function)
        graph.rpt = new idType[v_num + 1];
        graph.colids = new idType[e_num];
        graph.values = new valType[e_num];
        
        // Copy data to CSR structure
        std::copy(indptr.begin(), indptr.end(), graph.rpt);
        std::copy(indices.begin(), indices.end(), graph.colids);
        std::copy(values.begin(), values.end(), graph.values);
        
        return graph;
        
    } catch (const std::exception& e) {
        std::cerr << "Error loading graph " << graph_name << ": " << e.what() << std::endl;
        throw;
    }
}

// Dataset mapping functions
CSR<int, double> load_reddit_graph(const std::string& data_path) {
    return load_graph_from_spgemm_pruning<int, double>(data_path, "reddit");
}

CSR<int, double> load_flickr_graph(const std::string& data_path) {
    return load_graph_from_spgemm_pruning<int, double>(data_path, "flickr");
}

CSR<int, double> load_yelp_graph(const std::string& data_path) {
    return load_graph_from_spgemm_pruning<int, double>(data_path, "yelp");
}

CSR<int, double> load_ogbn_products_graph(const std::string& data_path) {
    return load_graph_from_spgemm_pruning<int, double>(data_path, "ogbn-products");
}

CSR<int, double> load_ogbn_proteins_graph(const std::string& data_path) {
    return load_graph_from_spgemm_pruning<int, double>(data_path, "ogbn-proteins");
}

#endif // GRAPH_LOADER_HPP
