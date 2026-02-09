"""
Big-O Complexity Cheatsheet
Apply to: Algorithm selection, performance interviews, capacity planning

Comprehensive reference for time and space complexity of data structures,
algorithms, database operations, and ML models.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class ComplexityClass(Enum):
    """Common complexity classes ranked by performance"""
    CONSTANT = "O(1)"
    LOGARITHMIC = "O(log n)"
    LINEAR = "O(n)"
    LINEARITHMIC = "O(n log n)"
    QUADRATIC = "O(n²)"
    CUBIC = "O(n³)"
    EXPONENTIAL = "O(2^n)"
    FACTORIAL = "O(n!)"
    SQRT = "O(√n)"
    N_LOG_LOG_N = "O(n log log n)"
    K_LINEAR = "O(k * n)"  # k is number of dimensions/features


@dataclass
class Complexity:
    """Represents time/space complexity for an operation"""
    best: str
    average: str
    worst: str
    space: str = "O(1)"
    
    def __str__(self):
        return f"Best: {self.best}, Avg: {self.average}, Worst: {self.worst}, Space: {self.space}"


@dataclass
class DataStructure:
    """Data structure with operation complexities"""
    name: str
    access: Complexity
    search: Complexity
    insert: Complexity
    delete: Complexity
    space: str
    when_to_use: str
    real_world_examples: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Algorithm:
    """Algorithm with complexity and use cases"""
    name: str
    time: Complexity
    space: str
    stable: bool = False
    when_to_use: str = ""
    real_world_examples: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class DatabaseOperation:
    """Database operation complexity"""
    name: str
    time_complexity: str
    space_complexity: str
    io_cost: str
    when_to_use: str
    real_world_examples: List[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class MLAlgorithm:
    """Machine learning algorithm complexity"""
    name: str
    training_time: str
    inference_time: str
    space: str
    when_to_use: str
    real_world_examples: List[str] = field(default_factory=list)
    notes: str = ""


# ============================================================================
# DATA STRUCTURES
# ============================================================================

DATA_STRUCTURES = {
    "array": DataStructure(
        name="Array (Dynamic Array)",
        access=Complexity("O(1)", "O(1)", "O(1)"),
        search=Complexity("O(1)", "O(n)", "O(n)"),
        insert=Complexity("O(1)", "O(n)", "O(n)", "O(n)"),
        delete=Complexity("O(1)", "O(n)", "O(n)"),
        space="O(n)",
        when_to_use="Random access, fixed-size collections, cache-friendly operations",
        real_world_examples=[
            "Image pixel data (width * height array)",
            "Time-series data (stock prices, sensor readings)",
            "LRU cache with array backing",
            "Video game entity lists",
            "Buffer for network packets"
        ],
        notes="Insertion/deletion at end is O(1) amortized. Resizing is O(n)."
    ),
    
    "linked_list": DataStructure(
        name="Linked List (Singly/Doubly)",
        access=Complexity("O(n)", "O(n)", "O(n)"),
        search=Complexity("O(1)", "O(n)", "O(n)"),
        insert=Complexity("O(1)", "O(1)", "O(1)"),
        delete=Complexity("O(1)", "O(1)", "O(1)"),
        space="O(n)",
        when_to_use="Frequent insertions/deletions at beginning, unknown size, queue/stack",
        real_world_examples=[
            "Browser history (back/forward navigation)",
            "Music playlist (prev/next track)",
            "Undo/redo functionality",
            "Memory allocation (free list)",
            "Blockchain structure"
        ],
        notes="Insert/delete assumes you have pointer to node. Double has 2x space overhead."
    ),
    
    "hash_map": DataStructure(
        name="Hash Map (Hash Table, Dictionary)",
        access=Complexity("N/A", "N/A", "N/A"),
        search=Complexity("O(1)", "O(1)", "O(n)"),
        insert=Complexity("O(1)", "O(1)", "O(n)"),
        delete=Complexity("O(1)", "O(1)", "O(n)"),
        space="O(n)",
        when_to_use="Fast lookups by key, caching, counting, deduplication",
        real_world_examples=[
            "Database indexes",
            "Session storage (user_id -> session data)",
            "DNS resolution cache",
            "Rate limiting (IP -> request count)",
            "Word frequency counter",
            "GraphQL DataLoader batching"
        ],
        notes="Worst case O(n) if all keys hash to same bucket. Load factor affects performance."
    ),
    
    "bst": DataStructure(
        name="Binary Search Tree (Balanced: AVL, Red-Black)",
        access=Complexity("O(log n)", "O(log n)", "O(n)"),
        search=Complexity("O(log n)", "O(log n)", "O(n)"),
        insert=Complexity("O(log n)", "O(log n)", "O(n)"),
        delete=Complexity("O(log n)", "O(log n)", "O(n)"),
        space="O(n)",
        when_to_use="Sorted data, range queries, ordered iteration",
        real_world_examples=[
            "Database indexes (B-tree variant)",
            "File system directories",
            "Priority-based task scheduling",
            "Auto-complete suggestions (trie alternative)",
            "Game leaderboards with rank queries"
        ],
        notes="Balanced BST guarantees O(log n). Unbalanced degrades to O(n)."
    ),
    
    "heap": DataStructure(
        name="Binary Heap (Min/Max Heap, Priority Queue)",
        access=Complexity("O(1)", "O(n)", "O(n)"),
        search=Complexity("O(n)", "O(n)", "O(n)"),
        insert=Complexity("O(log n)", "O(log n)", "O(log n)"),
        delete=Complexity("O(log n)", "O(log n)", "O(log n)"),
        space="O(n)",
        when_to_use="Priority queue, finding min/max, streaming median",
        real_world_examples=[
            "Dijkstra's shortest path",
            "Task scheduler (priority-based)",
            "Median of data stream",
            "Top K elements (heap of size K)",
            "Event-driven simulation",
            "Merge K sorted arrays"
        ],
        notes="Access is O(1) for min/max only. Full search is O(n)."
    ),
    
    "trie": DataStructure(
        name="Trie (Prefix Tree)",
        access=Complexity("O(k)", "O(k)", "O(k)"),
        search=Complexity("O(k)", "O(k)", "O(k)"),
        insert=Complexity("O(k)", "O(k)", "O(k)"),
        delete=Complexity("O(k)", "O(k)", "O(k)"),
        space="O(ALPHABET_SIZE * k * n)",
        when_to_use="Prefix matching, autocomplete, spell check, IP routing",
        real_world_examples=[
            "Search autocomplete (Google, IDE)",
            "Spell checker dictionary",
            "IP routing tables",
            "T9 predictive text",
            "DNA sequence analysis",
            "Contact name search"
        ],
        notes="k = key length. Space can be large but prefix sharing helps."
    ),
    
    "graph_adjacency_list": DataStructure(
        name="Graph (Adjacency List)",
        access=Complexity("O(1)", "O(1)", "O(1)"),
        search=Complexity("O(V+E)", "O(V+E)", "O(V+E)"),
        insert=Complexity("O(1)", "O(1)", "O(1)"),
        delete=Complexity("O(V+E)", "O(V+E)", "O(V+E)"),
        space="O(V+E)",
        when_to_use="Sparse graphs, typical for real-world networks",
        real_world_examples=[
            "Social networks (users and friendships)",
            "Web page links (PageRank)",
            "Road networks (GPS navigation)",
            "Dependency graphs (package managers)",
            "Neural network architectures"
        ],
        notes="V=vertices, E=edges. More space-efficient than adjacency matrix for sparse graphs."
    ),
    
    "graph_adjacency_matrix": DataStructure(
        name="Graph (Adjacency Matrix)",
        access=Complexity("O(1)", "O(1)", "O(1)"),
        search=Complexity("O(V²)", "O(V²)", "O(V²)"),
        insert=Complexity("O(1)", "O(1)", "O(1)"),
        delete=Complexity("O(1)", "O(1)", "O(1)"),
        space="O(V²)",
        when_to_use="Dense graphs, fast edge existence check",
        real_world_examples=[
            "Dense connection matrices",
            "Image adjacency (pixel neighbors)",
            "Fully connected neural networks",
            "Small dense graphs"
        ],
        notes="Better for dense graphs (E ≈ V²). Wastes space on sparse graphs."
    ),
    
    "bloom_filter": DataStructure(
        name="Bloom Filter",
        access=Complexity("N/A", "N/A", "N/A"),
        search=Complexity("O(k)", "O(k)", "O(k)"),
        insert=Complexity("O(k)", "O(k)", "O(k)"),
        delete=Complexity("N/A", "N/A", "N/A"),
        space="O(m)",
        when_to_use="Space-efficient set membership, can tolerate false positives",
        real_world_examples=[
            "Database query cache (avoid disk reads)",
            "Malicious URL detection",
            "Bitcoin wallet address validation",
            "Distributed systems (avoiding network calls)",
            "Spell checker (reduce dictionary lookups)"
        ],
        notes="k=hash functions, m=bit array size. False positives possible, no false negatives."
    ),
}


# ============================================================================
# SORTING ALGORITHMS
# ============================================================================

SORTING_ALGORITHMS = {
    "quick_sort": Algorithm(
        name="Quick Sort",
        time=Complexity("O(n log n)", "O(n log n)", "O(n²)", "O(log n)"),
        space="O(log n)",
        stable=False,
        when_to_use="General purpose, in-place sorting, average case matters",
        real_world_examples=[
            "C/C++ std::sort default",
            "Sorting large datasets in memory",
            "Database query result ordering",
            "File sorting utilities"
        ],
        notes="Worst case O(n²) with bad pivot selection. Use randomized pivot or median-of-3."
    ),
    
    "merge_sort": Algorithm(
        name="Merge Sort",
        time=Complexity("O(n log n)", "O(n log n)", "O(n log n)"),
        space="O(n)",
        stable=True,
        when_to_use="Stable sort needed, guaranteed O(n log n), external sorting",
        real_world_examples=[
            "Sorting linked lists (no random access needed)",
            "External sorting (disk-based data)",
            "Parallel sorting (divide and conquer)",
            "Java Collections.sort (for objects)"
        ],
        notes="Guaranteed O(n log n) but requires O(n) extra space."
    ),
    
    "heap_sort": Algorithm(
        name="Heap Sort",
        time=Complexity("O(n log n)", "O(n log n)", "O(n log n)"),
        space="O(1)",
        stable=False,
        when_to_use="Guaranteed O(n log n) with O(1) space, embedded systems",
        real_world_examples=[
            "Systems with limited memory",
            "Real-time systems (predictable performance)",
            "Priority queue implementation"
        ],
        notes="In-place but not stable. Worse cache performance than quicksort."
    ),
    
    "radix_sort": Algorithm(
        name="Radix Sort",
        time=Complexity("O(d*n)", "O(d*n)", "O(d*n)"),
        space="O(n+k)",
        stable=True,
        when_to_use="Sorting integers/strings with fixed length, linear time needed",
        real_world_examples=[
            "Sorting IP addresses",
            "Sorting strings of same length",
            "Sorting dates (YYYYMMDD)",
            "Parallel sorting of integers"
        ],
        notes="d=digits/characters, k=radix size. Not comparison-based, can beat O(n log n)."
    ),
    
    "tim_sort": Algorithm(
        name="Tim Sort",
        time=Complexity("O(n)", "O(n log n)", "O(n log n)"),
        space="O(n)",
        stable=True,
        when_to_use="Real-world data with partial ordering, Python/Java default",
        real_world_examples=[
            "Python sorted(), list.sort()",
            "Java Arrays.sort (for objects)",
            "Sorting partially sorted data",
            "General purpose stable sorting"
        ],
        notes="Hybrid of merge sort and insertion sort. Excellent on real-world data."
    ),
    
    "bubble_sort": Algorithm(
        name="Bubble Sort",
        time=Complexity("O(n)", "O(n²)", "O(n²)"),
        space="O(1)",
        stable=True,
        when_to_use="Educational purposes, nearly sorted data, tiny datasets",
        real_world_examples=[
            "Teaching sorting concepts",
            "Hardware sorting networks",
            "Tiny embedded systems (code simplicity)"
        ],
        notes="Simple but inefficient. Only use for tiny datasets or teaching."
    ),
    
    "insertion_sort": Algorithm(
        name="Insertion Sort",
        time=Complexity("O(n)", "O(n²)", "O(n²)"),
        space="O(1)",
        stable=True,
        when_to_use="Small datasets (<10 elements), nearly sorted, online sorting",
        real_world_examples=[
            "Sorting as data streams in (online)",
            "Final step in Tim Sort",
            "Small subarrays in quicksort",
            "Playing card sorting"
        ],
        notes="Efficient for small/nearly sorted data. Used in hybrid algorithms."
    ),
    
    "selection_sort": Algorithm(
        name="Selection Sort",
        time=Complexity("O(n²)", "O(n²)", "O(n²)"),
        space="O(1)",
        stable=False,
        when_to_use="Minimizing writes/swaps, educational purposes",
        real_world_examples=[
            "Flash memory sorting (minimize writes)",
            "Teaching algorithm concepts"
        ],
        notes="Always O(n²) but minimizes number of swaps to O(n)."
    ),
    
    "counting_sort": Algorithm(
        name="Counting Sort",
        time=Complexity("O(n+k)", "O(n+k)", "O(n+k)"),
        space="O(k)",
        stable=True,
        when_to_use="Small range of integers, need linear time",
        real_world_examples=[
            "Sorting ages (0-120)",
            "Sorting grades (A-F)",
            "Histogram generation",
            "Subroutine in radix sort"
        ],
        notes="k=range of input values. Not practical when k >> n."
    ),
}


# ============================================================================
# SEARCHING ALGORITHMS
# ============================================================================

SEARCHING_ALGORITHMS = {
    "binary_search": Algorithm(
        name="Binary Search",
        time=Complexity("O(1)", "O(log n)", "O(log n)"),
        space="O(1)",
        when_to_use="Sorted array, repeated searches, large dataset",
        real_world_examples=[
            "Dictionary lookup",
            "Database index search",
            "Git bisect (finding bug-introducing commit)",
            "Version search in sorted releases",
            "Finding insertion point"
        ],
        notes="Requires sorted data. Iterative is O(1) space, recursive is O(log n)."
    ),
    
    "linear_search": Algorithm(
        name="Linear Search",
        time=Complexity("O(1)", "O(n)", "O(n)"),
        space="O(1)",
        when_to_use="Unsorted data, small dataset, single search",
        real_world_examples=[
            "Finding element in small list",
            "Scanning log files",
            "Searching linked list",
            "First occurrence in unsorted data"
        ],
        notes="Simple but slow. Only option for unsorted data without preprocessing."
    ),
    
    "bfs": Algorithm(
        name="Breadth-First Search (BFS)",
        time=Complexity("O(V+E)", "O(V+E)", "O(V+E)"),
        space="O(V)",
        when_to_use="Shortest path (unweighted), level-order traversal, connected components",
        real_world_examples=[
            "Social network friend suggestions (degrees of separation)",
            "Web crawler (level-by-level)",
            "GPS navigation (unweighted roads)",
            "Network broadcast",
            "Puzzle solving (shortest moves)"
        ],
        notes="V=vertices, E=edges. Uses queue. Finds shortest path in unweighted graph."
    ),
    
    "dfs": Algorithm(
        name="Depth-First Search (DFS)",
        time=Complexity("O(V+E)", "O(V+E)", "O(V+E)"),
        space="O(V)",
        when_to_use="Topological sort, cycle detection, path finding, maze solving",
        real_world_examples=[
            "Maze solving",
            "Dependency resolution (build systems)",
            "Detecting cycles in graphs",
            "Generating permutations/combinations",
            "Backtracking problems (N-Queens, Sudoku)"
        ],
        notes="Uses stack (or recursion). Better space complexity for wide graphs."
    ),
    
    "dijkstra": Algorithm(
        name="Dijkstra's Algorithm",
        time=Complexity("O((V+E) log V)", "O((V+E) log V)", "O((V+E) log V)"),
        space="O(V)",
        when_to_use="Shortest path with non-negative weights, single source",
        real_world_examples=[
            "GPS navigation (fastest route)",
            "Network routing protocols (OSPF)",
            "Flight path optimization",
            "Robot pathfinding"
        ],
        notes="Requires non-negative weights. Use priority queue for efficiency."
    ),
    
    "a_star": Algorithm(
        name="A* Search",
        time=Complexity("O(E)", "O(E)", "O(b^d)"),
        space="O(b^d)",
        when_to_use="Shortest path with heuristic, game AI, robotics",
        real_world_examples=[
            "Game AI pathfinding (NPCs, units)",
            "Robot motion planning",
            "Route planning with heuristics",
            "Puzzle solving (15-puzzle, Rubik's cube)"
        ],
        notes="b=branching factor, d=depth. Performance depends on heuristic quality."
    ),
    
    "bellman_ford": Algorithm(
        name="Bellman-Ford Algorithm",
        time=Complexity("O(VE)", "O(VE)", "O(VE)"),
        space="O(V)",
        when_to_use="Shortest path with negative weights, detect negative cycles",
        real_world_examples=[
            "Currency arbitrage detection",
            "Network routing with cost penalties",
            "Distance vector routing protocols"
        ],
        notes="Slower than Dijkstra but handles negative weights. Detects negative cycles."
    ),
}


# ============================================================================
# DATABASE OPERATIONS
# ============================================================================

DATABASE_OPERATIONS = {
    "full_table_scan": DatabaseOperation(
        name="Full Table Scan (Sequential Scan)",
        time_complexity="O(n)",
        space_complexity="O(1)",
        io_cost="High - reads entire table",
        when_to_use="Small tables, no index available, returning most rows",
        real_world_examples=[
            "SELECT * FROM users (small table)",
            "Aggregate queries without index (COUNT, SUM)",
            "Queries without WHERE clause",
            "Full text search without index"
        ],
        notes="Scans every row. Avoid for large tables with selective queries."
    ),
    
    "index_scan": DatabaseOperation(
        name="Index Scan (B-tree Index)",
        time_complexity="O(log n + k)",
        space_complexity="O(log n)",
        io_cost="Low to Medium - reads index + matching rows",
        when_to_use="Selective queries, range queries, sorting",
        real_world_examples=[
            "SELECT * FROM users WHERE user_id = 123",
            "SELECT * FROM orders WHERE created_at BETWEEN ... AND ...",
            "SELECT * FROM products ORDER BY price",
            "JOIN operations on indexed columns"
        ],
        notes="k=matching rows. B-tree index: O(log n) for search, supports range queries."
    ),
    
    "hash_index_lookup": DatabaseOperation(
        name="Hash Index Lookup",
        time_complexity="O(1)",
        space_complexity="O(1)",
        io_cost="Very Low - direct access",
        when_to_use="Equality lookups, unique key access",
        real_world_examples=[
            "SELECT * FROM users WHERE email = 'user@example.com'",
            "Primary key lookups",
            "Exact match queries"
        ],
        notes="Only supports equality (=). No range queries or sorting."
    ),
    
    "nested_loop_join": DatabaseOperation(
        name="Nested Loop Join",
        time_complexity="O(n * m)",
        space_complexity="O(1)",
        io_cost="Very High - reads outer table fully, inner per row",
        when_to_use="Small outer table, inner table has index on join key",
        real_world_examples=[
            "Small dimension table joined to fact table",
            "JOIN when one table has very few rows"
        ],
        notes="n=outer table rows, m=inner table rows. Avoid for large tables."
    ),
    
    "hash_join": DatabaseOperation(
        name="Hash Join",
        time_complexity="O(n + m)",
        space_complexity="O(min(n, m))",
        io_cost="Medium - reads both tables once",
        when_to_use="Equi-joins on large tables, no suitable index",
        real_world_examples=[
            "SELECT * FROM orders o JOIN customers c ON o.customer_id = c.id",
            "Data warehouse aggregations",
            "Large table joins without indexes"
        ],
        notes="Builds hash table of smaller table. Best for equality joins."
    ),
    
    "sort_merge_join": DatabaseOperation(
        name="Sort-Merge Join",
        time_complexity="O(n log n + m log m)",
        space_complexity="O(n + m)",
        io_cost="High - sorting overhead",
        when_to_use="Both inputs already sorted, range joins",
        real_world_examples=[
            "JOIN on sorted data (indexed columns)",
            "Range joins (BETWEEN)",
            "Non-equi joins"
        ],
        notes="Efficient if inputs already sorted. Works for non-equality joins."
    ),
    
    "index_only_scan": DatabaseOperation(
        name="Index-Only Scan (Covering Index)",
        time_complexity="O(log n + k)",
        space_complexity="O(log n)",
        io_cost="Very Low - only reads index",
        when_to_use="All query columns in index, no table access needed",
        real_world_examples=[
            "SELECT user_id FROM users WHERE email = 'x' (both in index)",
            "SELECT COUNT(*) with index",
            "Queries only accessing indexed columns"
        ],
        notes="Fastest option. Index must include all queried columns."
    ),
    
    "bitmap_index_scan": DatabaseOperation(
        name="Bitmap Index Scan",
        time_complexity="O(n)",
        space_complexity="O(n)",
        io_cost="Low to Medium",
        when_to_use="Low cardinality columns, multiple OR conditions, data warehouses",
        real_world_examples=[
            "SELECT * FROM users WHERE status IN ('active', 'pending')",
            "Gender, country, boolean flags",
            "Data warehouse star schema dimensions"
        ],
        notes="Efficient for columns with few distinct values. Combines multiple conditions well."
    ),
}


# ============================================================================
# MACHINE LEARNING ALGORITHMS
# ============================================================================

ML_ALGORITHMS = {
    "linear_regression": MLAlgorithm(
        name="Linear Regression",
        training_time="O(n * d²) or O(n * d) with SGD",
        inference_time="O(d)",
        space="O(d)",
        when_to_use="Continuous prediction, interpretable model, linear relationships",
        real_world_examples=[
            "Housing price prediction",
            "Sales forecasting",
            "Risk scoring (credit)",
            "Demand prediction"
        ],
        notes="n=samples, d=features. Fast training and inference. Assumes linearity."
    ),
    
    "logistic_regression": MLAlgorithm(
        name="Logistic Regression",
        training_time="O(n * d) with SGD",
        inference_time="O(d)",
        space="O(d)",
        when_to_use="Binary/multi-class classification, baseline model, interpretable",
        real_world_examples=[
            "Spam detection",
            "Click-through rate prediction",
            "Medical diagnosis (disease/no disease)",
            "Customer churn prediction"
        ],
        notes="Fast and interpretable. Works well with high-dimensional sparse data."
    ),
    
    "decision_tree": MLAlgorithm(
        name="Decision Tree",
        training_time="O(n * d * log n)",
        inference_time="O(log n)",
        space="O(n)",
        when_to_use="Non-linear patterns, interpretability, mixed data types",
        real_world_examples=[
            "Credit approval (explainable decisions)",
            "Medical diagnosis trees",
            "Customer segmentation",
            "Rule-based systems"
        ],
        notes="Prone to overfitting. Use ensemble methods (RF, GBM) for better results."
    ),
    
    "random_forest": MLAlgorithm(
        name="Random Forest",
        training_time="O(n * d * log n * k)",
        inference_time="O(k * log n)",
        space="O(k * n)",
        when_to_use="Robust predictions, feature importance, handles overfitting",
        real_world_examples=[
            "Fraud detection",
            "Recommendation systems",
            "Sensor data classification",
            "Kaggle competitions (baseline)"
        ],
        notes="k=number of trees. Less prone to overfitting than single tree."
    ),
    
    "gradient_boosting": MLAlgorithm(
        name="Gradient Boosting (XGBoost, LightGBM)",
        training_time="O(n * d * log n * k)",
        inference_time="O(k * log n)",
        space="O(k * n)",
        when_to_use="High accuracy needed, tabular data, feature interactions",
        real_world_examples=[
            "Kaggle winning solutions",
            "Click-through rate prediction",
            "Learning to rank (search)",
            "Risk modeling"
        ],
        notes="Often best for tabular data. Slower training than RF. Can overfit."
    ),
    
    "svm": MLAlgorithm(
        name="Support Vector Machine (SVM)",
        training_time="O(n² * d) to O(n³ * d)",
        inference_time="O(k * d)",
        space="O(k * d)",
        when_to_use="Small to medium datasets, high-dimensional data, kernel methods",
        real_world_examples=[
            "Text classification",
            "Image classification (with kernels)",
            "Bioinformatics",
            "Handwriting recognition"
        ],
        notes="k=support vectors. Slow training on large datasets. Kernel trick powerful."
    ),
    
    "knn": MLAlgorithm(
        name="K-Nearest Neighbors (KNN)",
        training_time="O(1)",
        inference_time="O(n * d)",
        space="O(n * d)",
        when_to_use="Small datasets, no training time, non-parametric",
        real_world_examples=[
            "Recommendation systems (item similarity)",
            "Anomaly detection",
            "Missing value imputation",
            "Pattern recognition"
        ],
        notes="Lazy learning (no training). Slow inference. Sensitive to feature scaling."
    ),
    
    "kmeans": MLAlgorithm(
        name="K-Means Clustering",
        training_time="O(n * k * d * i)",
        inference_time="O(k * d)",
        space="O(k * d)",
        when_to_use="Customer segmentation, data compression, initialization",
        real_world_examples=[
            "Customer segmentation",
            "Image compression (color quantization)",
            "Document clustering",
            "Anomaly detection (distance to nearest centroid)"
        ],
        notes="i=iterations. Sensitive to initialization. Assumes spherical clusters."
    ),
    
    "naive_bayes": MLAlgorithm(
        name="Naive Bayes",
        training_time="O(n * d)",
        inference_time="O(c * d)",
        space="O(c * d)",
        when_to_use="Text classification, spam filtering, fast baseline",
        real_world_examples=[
            "Email spam detection",
            "Sentiment analysis",
            "Document categorization",
            "Real-time prediction (very fast)"
        ],
        notes="c=classes. Very fast. Assumes feature independence (often violated)."
    ),
    
    "neural_network_dense": MLAlgorithm(
        name="Neural Network (Dense/Fully Connected)",
        training_time="O(n * d * h * e)",
        inference_time="O(d * h)",
        space="O(d * h + h²)",
        when_to_use="Complex patterns, large datasets, sufficient compute",
        real_world_examples=[
            "Tabular data with complex interactions",
            "Time series prediction",
            "Multi-task learning",
            "Embedding generation"
        ],
        notes="h=hidden units, e=epochs. Requires tuning. Can overfit without regularization."
    ),
    
    "cnn": MLAlgorithm(
        name="Convolutional Neural Network (CNN)",
        training_time="O(n * w * h * c * k * e)",
        inference_time="O(w * h * c * k)",
        space="O(w * h * c * k)",
        when_to_use="Image/video processing, spatial patterns, computer vision",
        real_world_examples=[
            "Image classification (ImageNet)",
            "Object detection (YOLO, Faster R-CNN)",
            "Face recognition",
            "Medical image analysis",
            "Autonomous driving (vision)"
        ],
        notes="w,h=image dimensions, c=channels, k=kernels. GPU accelerated."
    ),
    
    "rnn_lstm": MLAlgorithm(
        name="RNN/LSTM/GRU",
        training_time="O(n * t * d * h * e)",
        inference_time="O(t * d * h)",
        space="O(t * h)",
        when_to_use="Sequential data, time series, variable-length input",
        real_world_examples=[
            "Language modeling",
            "Machine translation (legacy, pre-Transformer)",
            "Speech recognition",
            "Time series prediction",
            "Video analysis"
        ],
        notes="t=sequence length, h=hidden size. Sequential computation limits parallelism."
    ),
    
    "transformer": MLAlgorithm(
        name="Transformer (BERT, GPT, T5)",
        training_time="O(n * t² * d * e)",
        inference_time="O(t² * d)",
        space="O(t² + t * d)",
        when_to_use="NLP tasks, long-range dependencies, parallel training",
        real_world_examples=[
            "Language models (GPT-4, ChatGPT)",
            "Machine translation",
            "Question answering",
            "Text summarization",
            "Code generation (Copilot)"
        ],
        notes="t=sequence length. Quadratic in sequence length. Highly parallelizable."
    ),
    
    "pca": MLAlgorithm(
        name="Principal Component Analysis (PCA)",
        training_time="O(n * d² + d³)",
        inference_time="O(d * k)",
        space="O(d²)",
        when_to_use="Dimensionality reduction, visualization, noise reduction",
        real_world_examples=[
            "Data visualization (reduce to 2D/3D)",
            "Feature extraction",
            "Image compression",
            "Noise reduction in signals"
        ],
        notes="k=components. Linear method. Loses interpretability of features."
    ),
}


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_complexity(category: str, name: str) -> dict:
    """
    Look up complexity information for a specific algorithm or data structure.
    
    Args:
        category: One of 'data_structure', 'sorting', 'searching', 'database', 'ml'
        name: Name of the algorithm or data structure
        
    Returns:
        Dict with complexity information
    """
    categories = {
        'data_structure': DATA_STRUCTURES,
        'sorting': SORTING_ALGORITHMS,
        'searching': SEARCHING_ALGORITHMS,
        'database': DATABASE_OPERATIONS,
        'ml': ML_ALGORITHMS
    }
    
    if category not in categories:
        return {"error": f"Unknown category: {category}"}
    
    item = categories[category].get(name)
    if not item:
        return {"error": f"Unknown {category}: {name}"}
    
    return vars(item)


def compare_algorithms(category: str, names: List[str], operation: str = None) -> None:
    """
    Compare time complexity of multiple algorithms.
    
    Args:
        category: One of 'sorting', 'searching', 'ml'
        names: List of algorithm names to compare
        operation: For data structures, specify operation (access, search, insert, delete)
    """
    print(f"\n{'='*80}")
    print(f"COMPARISON: {category.upper()}")
    print(f"{'='*80}\n")
    
    categories = {
        'data_structure': DATA_STRUCTURES,
        'sorting': SORTING_ALGORITHMS,
        'searching': SEARCHING_ALGORITHMS,
        'database': DATABASE_OPERATIONS,
        'ml': ML_ALGORITHMS
    }
    
    if category not in categories:
        print(f"Error: Unknown category: {category}")
        return
    
    items = categories[category]
    
    for name in names:
        if name not in items:
            print(f"Warning: {name} not found in {category}")
            continue
            
        item = items[name]
        print(f"{item.name}:")
        
        if category == 'data_structure' and operation:
            op_complexity = getattr(item, operation, None)
            if op_complexity:
                print(f"  {operation.capitalize()}: {op_complexity}")
        elif category in ['sorting', 'searching']:
            print(f"  Time: {item.time}")
            print(f"  Space: {item.space}")
        elif category == 'database':
            print(f"  Time: {item.time_complexity}")
            print(f"  I/O Cost: {item.io_cost}")
        elif category == 'ml':
            print(f"  Training: {item.training_time}")
            print(f"  Inference: {item.inference_time}")
        
        print()


def find_best_for_use_case(use_case: str) -> List[str]:
    """
    Search for algorithms/data structures matching a use case.
    
    Args:
        use_case: Description of the problem (e.g., "sorted data", "graph", "text")
        
    Returns:
        List of matching algorithm/data structure names with categories
    """
    results = []
    use_case_lower = use_case.lower()
    
    all_items = [
        ('data_structure', DATA_STRUCTURES),
        ('sorting', SORTING_ALGORITHMS),
        ('searching', SEARCHING_ALGORITHMS),
        ('database', DATABASE_OPERATIONS),
        ('ml', ML_ALGORITHMS)
    ]
    
    for category_name, category_dict in all_items:
        for key, item in category_dict.items():
            when_to_use = item.when_to_use.lower() if hasattr(item, 'when_to_use') else ""
            examples = " ".join(item.real_world_examples).lower() if hasattr(item, 'real_world_examples') else ""
            notes = item.notes.lower() if hasattr(item, 'notes') else ""
            
            if (use_case_lower in when_to_use or 
                use_case_lower in examples or 
                use_case_lower in notes):
                results.append(f"{category_name}: {item.name}")
    
    return results


def print_category(category: str, filter_name: str = None) -> None:
    """
    Pretty print all items in a category or a specific item.
    
    Args:
        category: One of 'data_structure', 'sorting', 'searching', 'database', 'ml'
        filter_name: Optional name to filter to a specific item
    """
    categories = {
        'data_structure': DATA_STRUCTURES,
        'sorting': SORTING_ALGORITHMS,
        'searching': SEARCHING_ALGORITHMS,
        'database': DATABASE_OPERATIONS,
        'ml': ML_ALGORITHMS
    }
    
    if category not in categories:
        print(f"Error: Unknown category: {category}")
        return
    
    items = categories[category]
    
    print(f"\n{'='*80}")
    print(f"{category.upper().replace('_', ' ')}")
    print(f"{'='*80}\n")
    
    for key, item in items.items():
        if filter_name and key != filter_name:
            continue
            
        print(f"📊 {item.name}")
        print(f"{'─'*80}")
        
        if isinstance(item, DataStructure):
            print(f"  Access:  {item.access}")
            print(f"  Search:  {item.search}")
            print(f"  Insert:  {item.insert}")
            print(f"  Delete:  {item.delete}")
            print(f"  Space:   {item.space}")
        elif isinstance(item, Algorithm):
            print(f"  Time:    {item.time}")
            print(f"  Space:   {item.space}")
            if hasattr(item, 'stable'):
                print(f"  Stable:  {item.stable}")
        elif isinstance(item, DatabaseOperation):
            print(f"  Time:    {item.time_complexity}")
            print(f"  Space:   {item.space_complexity}")
            print(f"  I/O:     {item.io_cost}")
        elif isinstance(item, MLAlgorithm):
            print(f"  Training:   {item.training_time}")
            print(f"  Inference:  {item.inference_time}")
            print(f"  Space:      {item.space}")
        
        print(f"\n  ✓ When to use:")
        print(f"    {item.when_to_use}")
        
        if hasattr(item, 'real_world_examples') and item.real_world_examples:
            print(f"\n  💡 Real-world examples:")
            for example in item.real_world_examples[:3]:
                print(f"    • {example}")
        
        if hasattr(item, 'notes') and item.notes:
            print(f"\n  📝 Notes: {item.notes}")
        
        print("\n")


def print_complexity_guide() -> None:
    """Print a visual guide to Big-O complexity classes."""
    print("\n" + "="*80)
    print("BIG-O COMPLEXITY REFERENCE")
    print("="*80 + "\n")
    
    print("Excellent → Good → Fair → Bad → Horrible\n")
    
    complexities = [
        ("O(1)", "Constant", "✓✓✓ Excellent", "Hash table lookup, array access"),
        ("O(log n)", "Logarithmic", "✓✓✓ Excellent", "Binary search, balanced tree ops"),
        ("O(n)", "Linear", "✓✓ Good", "Linear search, single loop"),
        ("O(n log n)", "Linearithmic", "✓✓ Good", "Efficient sorting (merge, heap, quick)"),
        ("O(n²)", "Quadratic", "✓ Fair", "Nested loops, bubble sort"),
        ("O(n³)", "Cubic", "✗ Bad", "Triple nested loops"),
        ("O(2^n)", "Exponential", "✗✗ Horrible", "Recursive fibonacci, subset generation"),
        ("O(n!)", "Factorial", "✗✗✗ Horrible", "Permutations, traveling salesman"),
    ]
    
    for notation, name, rating, example in complexities:
        print(f"{notation:12} {name:15} {rating:20} {example}")
    
    print("\n" + "="*80)
    print("GROWTH COMPARISON (n = 1000)")
    print("="*80 + "\n")
    
    n = 1000
    print(f"O(1):        {1:>20,} operations")
    print(f"O(log n):    {10:>20,} operations")
    print(f"O(n):        {n:>20,} operations")
    print(f"O(n log n):  {n * 10:>20,} operations")
    print(f"O(n²):       {n**2:>20,} operations")
    print(f"O(n³):       {n**3:>20,} operations")
    print(f"O(2^n):      {'>10^300':>20} operations (intractable)")
    print(f"O(n!):       {'>10^2500':>20} operations (intractable)")
    
    print("\n")


def print_interview_guide() -> None:
    """Print common complexity patterns for interviews."""
    print("\n" + "="*80)
    print("INTERVIEW COMPLEXITY PATTERNS")
    print("="*80 + "\n")
    
    patterns = [
        ("Single loop over n items", "O(n)"),
        ("Nested loops over n items", "O(n²)"),
        ("Binary search / halving", "O(log n)"),
        ("Sorting then processing", "O(n log n)"),
        ("Recursion with branching (tree)", "O(2^n) or O(branches^depth)"),
        ("Hash table operations", "O(1) average, O(n) worst"),
        ("Two pointers (sorted array)", "O(n)"),
        ("Sliding window", "O(n)"),
        ("DFS/BFS on graph", "O(V + E)"),
        ("Dynamic programming (memoization)", "O(states × transition)"),
    ]
    
    for pattern, complexity in patterns:
        print(f"{pattern:40} → {complexity}")
    
    print("\n")


# ============================================================================
# EXAMPLES AND USAGE
# ============================================================================

def main():
    """Run examples of using the complexity cheatsheet."""
    
    # Print complexity guide
    print_complexity_guide()
    
    # Print interview patterns
    print_interview_guide()
    
    # Example 1: Look up a specific data structure
    print("\n" + "="*80)
    print("EXAMPLE 1: Look up hash map complexity")
    print("="*80)
    result = get_complexity('data_structure', 'hash_map')
    print(f"Name: {result['name']}")
    print(f"Search: {result['search']}")
    print(f"When to use: {result['when_to_use']}")
    
    # Example 2: Compare sorting algorithms
    print("\n" + "="*80)
    print("EXAMPLE 2: Compare sorting algorithms")
    print("="*80)
    compare_algorithms('sorting', ['quick_sort', 'merge_sort', 'tim_sort'])
    
    # Example 3: Find algorithms for a use case
    print("\n" + "="*80)
    print("EXAMPLE 3: Find algorithms for 'graph' use case")
    print("="*80)
    results = find_best_for_use_case('graph')
    for result in results:
        print(f"  • {result}")
    
    # Example 4: Print specific category
    print_category('data_structure', 'hash_map')
    
    # Example 5: Print all ML algorithms
    print_category('ml')


if __name__ == "__main__":
    main()
