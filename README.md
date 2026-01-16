# GRADWIZARD - A modern C++ engine for automatic differentiation

This project is a **C++ implementation of a micrograd-style automatic differentiation engine**, inspired by Andrej Karpathy’s micrograd.  
The focus is on **understanding gradwizard internals**, correct **ownership semantics**, and building a **clean multi-file C++ project** from first principles.

---

## Directory Structure

engine/
|
|---> NN.hpp // Declarations for Neuron, Layer, MLP
|---> NN.cpp // Definitions for Neuron, Layer, MLP
|
|---> gradwizard.hpp // Declarations for node, gradwizard logic, print_tree
|---> gradwizard.cpp // Definitions for node, DFS backward, print_tree
|
|---> rng.hpp // Random number generator interface
|---> rng.cpp // RNG implementation (mt19937, uniform distribution)
|
|---> main.cpp // Entry point: builds graph, runs forward/backward
|
|---> output.txt // Sample output / debugging logs
|---> temp.cpp // Scratch / experimentation file
|
|---> .gitignore
|---> README.md

---

## Core Design Overview

The engine is divided into two major layers:

1. **gradwizard Core (`node`)**
2. **Neural Network Abstractions (`Neuron`, `Layer`, `MLP`)**

---

## 1. `node` Class (gradwizard Core)

### Purpose

`node` represents a **single scalar value** in the computation graph.  
Each node stores:

- forward value (`data`)
- accumulated gradient (`grad`)
- parent nodes (`parents`)
- local backward function (`_backward`)
- operation label (`op`) and debug label (`label`)

---

### Why `shared_ptr<node>` is used

The computation graph is a **directed acyclic graph (DAG)** where:
- multiple nodes may reference the same parent
- node lifetimes extend beyond local scopes
- ownership is shared across the graph

Using `shared_ptr<node>` ensures:
- safe memory management
- no dangling references
- nodes remain alive as long as they are part of the graph

---

## 2. Backpropagation via DFS (Single `backward()` Call)

### Problem

Naively calling `_backward()` on nodes can:
- execute nodes multiple times
- violate topological order
- produce incorrect gradients

---

### Solution: Depth-First Search (DFS)

A DFS is used to:
1. Traverse the graph from the output node
2. Build a topological ordering
3. Execute `_backward()` **exactly once per node**, in reverse order

---

### Algorithm (Conceptual)

dfs(node):
    if node not visited:
        mark visited
    for parent in node.prev:
        dfs(parent)
    push node into topo list

backward():
    topo = []
    dfs(output)
    output.grad = 1
    for node in reverse(topo):
        node._backward()

This guarantees:
- correct gradient flow
- no duplicate backward calls
- proper chain-rule application

---

## 3. Neural Network Abstractions (`NN.hpp / NN.cpp`)

### `Neuron`

- Holds:
  - weights: `vector<shared_ptr<node>>`
  - bias: `shared_ptr<node>`
- Forward pass:
  - computes weighted sum
  - applies activation (e.g. `tanh`)
- Returns a **single `shared_ptr<node>`**

---

### `Layer`

- A collection of `Neuron`
- Forward pass maps:
- `parameters()` flattens all neuron parameters

---

### `MLP`

- A sequence of `Layer`
- Forward pass chains layers
- `parameters()` returns **all trainable nodes**

---

### Ownership Model

MLP
|---> Layer
        |---> Neuron
                |---> shared_ptr<node>

---

## 4. `print_tree()` – Computation Graph Visualization

### Purpose

`print_tree()` provides a **human-readable visualization** of the computation graph.

---

### Features

- DFS traversal
- `visited` set to prevent infinite loops
- Optional annotations:
  - value
  - gradient
  - operation
  - label

---

### Why it matters

- Makes the computation graph visible
- Helps debug incorrect gradients
- Confirms graph structure matches expectations

---

## 5. Random Number Generator (`rng.hpp / rng.cpp`)

- Uses `std::mt19937`
- Uniform real distribution
- Centralized RNG design avoids:
  - multiple seeds
  - hidden globals in headers
  - inconsistent initialization

This ensures reproducible weight initialization.

---

# How to Build and Run this engine

---

## 1. Clone the Repository

Open terminal and write : 

git clone https://github.com/teradadacodez/gradwizard_cpp
cd gradwizard_cpp
cd engine
g++ *.cpp -o app
.\app.exe (for windows 10+) || app (for linux)

=========================================================================================