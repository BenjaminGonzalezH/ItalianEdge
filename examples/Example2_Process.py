"""
Example2_Process.py

Standalone validation script for the principal mathematical functions
in the clustering source code.

Each section walks through one function as an explicit
    input -> function -> output pipeline,
with a step-by-step theoretical derivation and a human-verifiable
expected value computed by hand.

Functions covered
-----------------
1. consensus_matrix             (consensus_matrix.py)
2. jaccard_index_solutions      (jaccard_values.py)
3. jaccard_index_clusters       (jaccard_values.py)
4. rand_index_clusters          (rand_values.py -- _rand_from_binary_contingency)
5. adjusted_rand_index_clusters (rand_values.py -- _ari_from_binary_contingency)
6. _align_partition             (plurality_voting.py)
"""

import numpy as np

import gclusters_characterization.clustering.consensus_matrix as CM
import gclusters_characterization.clustering.jaccard_values as JV
import gclusters_characterization.clustering.rand_values as RV


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _header(name):
    print()
    print("=" * 56)
    print("FUNCTION: " + name)
    print("=" * 56)

def _section(label):
    print()
    print(label + ":")

def _result(result_val, expected_val):
    match = np.allclose(result_val, expected_val, atol=1e-9)
    print("  Result   : " + str(result_val))
    print("  Expected : " + str(expected_val))
    print("  Match    : " + str(match))
    print("=" * 56)


# ======================================================
# 1. consensus_matrix
# ======================================================

_header("consensus_matrix")

_section("Theory")
print("  The consensus matrix summarises how consistently two")
print("  elements co-cluster across multiple solutions.")
print()
print("  Step 1: For each solution, group elements by cluster label.")
print("  Step 2: For every pair (i, j) that share a cluster,")
print("          increment the coincidence accumulator C[i,j].")
print("  Step 3: After all solutions, divide C by the number of")
print("          solutions  ->  C[i,j] in [0, 1].")
print("  Step 4: Set the diagonal C[i,i] = 1 (self-similarity).")
print("  Step 5: consensus = 1 - C  (distance form).")

_section("Input")
print("  3 solutions, 4 genes (0-3).")
print()
print("  Solution 0 : [1, 1, 2, 2]  -> cluster 1: {0,1}   cluster 2: {2,3}")
print("  Solution 1 : [1, 1, 2, 2]  -> cluster 1: {0,1}   cluster 2: {2,3}")
print("  Solution 2 : [2, 2, 1, 1]  -> cluster 2: {0,1}   cluster 1: {2,3}")
print()
print("  Pair (0,1): co-cluster in all 3 solutions -> C[0,1] = 3/3 = 1.0")
print("  Pair (2,3): co-cluster in all 3 solutions -> C[2,3] = 3/3 = 1.0")
print("  Pair (0,2): never co-cluster              -> C[0,2] = 0/3 = 0.0")
print("  (symmetric; diagonal forced to 1.0)")
print()
print("  Expected coincidence matrix:")
print("    [[1, 1, 0, 0],")
print("     [1, 1, 0, 0],")
print("     [0, 0, 1, 1],")
print("     [0, 0, 1, 1]]")

Solutions = np.array([
    [1, 1, 2, 2],
    [1, 1, 2, 2],
    [2, 2, 1, 1],
])

coincidence, consensus = CM.consensus_matrix(Solutions)

expected_coincidence = np.array([
    [1., 1., 0., 0.],
    [1., 1., 0., 0.],
    [0., 0., 1., 1.],
    [0., 0., 1., 1.],
])

_section("Output  [coincidence matrix]")
_result(coincidence, expected_coincidence)


# ======================================================
# 2. jaccard_index_solutions
# ======================================================

_header("jaccard_index_solutions")

_section("Theory")
print("  The solution-level Jaccard index compares two clustering")
print("  solutions by treating each one as a binary co-membership")
print("  vector over all gene pairs.")
print()
print("  Step 1: For every pair of genes (i,j), build a Boolean")
print("          vector:  v[pair] = 1 if genes share a cluster, else 0.")
print("  Step 2: Given vectors v1 and v2 for two solutions, count:")
print("            r = |v1 AND v2|        (both agree: same cluster)")
print("            u = |v1 AND NOT v2|    (v1 says same, v2 says diff)")
print("            v = |NOT v1 AND v2|    (v1 says diff, v2 says same)")
print("  Step 3: J = r / (r + u + v).")

_section("Input")
print("  2 solutions, 5 genes (0-4).")
print()
print("  Solution 0 : [1, 1, 1, 2, 2]")
print("  Solution 1 : [1, 1, 2, 2, 2]")
print()
print("  C(5,2) = 10 gene pairs.")
print()
print("  v0 (Sol 0 co-membership per pair):")
print("    (0,1)T (0,2)T (0,3)F (0,4)F (1,2)T (1,3)F (1,4)F (2,3)F (2,4)F (3,4)T")
print()
print("  v1 (Sol 1 co-membership per pair):")
print("    (0,1)T (0,2)F (0,3)F (0,4)F (1,2)F (1,3)F (1,4)F (2,3)T (2,4)T (3,4)T")
print()
print("  r = |(0,1), (3,4)| = 2  [both True in v0 and v1]")
print("  u = |(0,2), (1,2)| = 2  [True in v0, False in v1]")
print("  v = |(2,3), (2,4)| = 2  [False in v0, True in v1]")
print()
print("  J(Sol0, Sol1) = 2 / (2+2+2) = 2/6 = 0.3333")

Solutions2 = np.array([
    [1, 1, 1, 2, 2],
    [1, 1, 2, 2, 2],
])

J_sol = JV.jaccard_index_solutions(Solutions2)

expected_J_sol = np.array([
    [1.0,  1/3],
    [1/3,  1.0],
])

_section("Output  [Jaccard similarity matrix, off-diagonal = J(Sol0,Sol1)]")
_result(J_sol, expected_J_sol)


# ======================================================
# 3. jaccard_index_clusters
# ======================================================

_header("jaccard_index_clusters")

_section("Theory")
print("  The cluster-level Jaccard index compares every cluster from")
print("  solution 1 against every cluster from solution 2 directly")
print("  using set operations.")
print()
print("  Step 1: For clusters A and B represented as sets of genes,")
print("          compute:  intersection = elements in both A and B")
print("                    union        = elements in A or B")
print("  Step 2: J(A, B) = |intersection| / |union|.")
print("  Step 3: Repeat for every (i, j) cluster pair across solutions.")

_section("Input")
print("  Solution 1 (2 clusters): A0 = {0,1,2}   A1 = {3,4}")
print("  Solution 2 (2 clusters): B0 = {0,1}     B1 = {2,3,4}")
print()
print("  J(A0, B0): |{0,1,2} & {0,1}| / |{0,1,2} | {0,1}|  = 2/3  ~ 0.667")
print("  J(A0, B1): |{0,1,2} & {2,3,4}| / |{0,1,2} | {2,3,4}| = 1/5 = 0.200")
print("  J(A1, B0): |{3,4}   & {0,1}|   / |{3,4}   | {0,1}|   = 0/4 = 0.000")
print("  J(A1, B1): |{3,4}   & {2,3,4}| / |{3,4}   | {2,3,4}| = 2/3 ~ 0.667")

Sol1_clusters = [{0, 1, 2}, {3, 4}]
Sol2_clusters = [{0, 1}, {2, 3, 4}]

J_clust = JV.jaccard_index_clusters(Sol1_clusters, Sol2_clusters)

expected_J_clust = np.array([
    [2/3,  1/5],
    [0.0,  2/3],
])

_section("Output  [2x2 cluster Jaccard matrix]")
_result(J_clust, expected_J_clust)


# ======================================================
# 4. rand_index_clusters
# ======================================================

_header("rand_index_clusters  [_rand_from_binary_contingency]")

_section("Theory")
print("  The Rand Index between two clusters A and B is computed over")
print("  the union U = A | B (elements seen by at least one cluster).")
print()
print("  Step 1: Build the binary contingency over U:")
print("            n11 = |A & B|           (in both)")
print("            n10 = |A| - n11         (in A only)")
print("            n01 = |B| - n11         (in B only)")
print("            n   = n11 + n10 + n01   (union size)")
print("  Step 2: Count agreeing pairs over all C(n,2) element pairs:")
print("            same_same = C(n11, 2)   (both assign pair to same cluster)")
print("            diff_diff = n10 * n01   (both assign pair to diff clusters)")
print("            agreements = same_same + diff_diff")
print("  Step 3: RI = agreements / C(n, 2).")
print("          When C(n,2) = 0 (singleton union), RI = 1 by convention.")

_section("Input")
print("  Solution 1 (2 clusters): A0 = {0,1,2}   A1 = {3,4}")
print("  Solution 2 (2 clusters): B0 = {0,1}     B1 = {2,3,4}")
print()
print("  Pair (A0, B0): n11=2, n10=1, n01=0, n=3, C(3,2)=3")
print("    same_same = C(2,2) = 1   diff_diff = 1*0 = 0")
print("    RI = 1/3 ~ 0.333")
print()
print("  Pair (A0, B1): n11=1, n10=2, n01=2, n=5, C(5,2)=10")
print("    same_same = C(1,2) = 0   diff_diff = 2*2 = 4")
print("    RI = 4/10 = 0.400")
print()
print("  Pair (A1, B0): n11=0, n10=2, n01=2, n=4, C(4,2)=6")
print("    same_same = C(0,2) = 0   diff_diff = 2*2 = 4")
print("    RI = 4/6 ~ 0.667")
print()
print("  Pair (A1, B1): n11=2, n10=0, n01=1, n=3, C(3,2)=3")
print("    same_same = C(2,2) = 1   diff_diff = 0*1 = 0")
print("    RI = 1/3 ~ 0.333")

RI_clust = RV.rand_index_clusters(Sol1_clusters, Sol2_clusters)

expected_RI_clust = np.array([
    [1/3,   4/10],
    [4/6,   1/3 ],
])

_section("Output  [2x2 cluster RI matrix]")
_result(RI_clust, expected_RI_clust)


# ======================================================
# 5. adjusted_rand_index_clusters
# ======================================================

_header("adjusted_rand_index_clusters  [_ari_from_binary_contingency]")

_section("Theory")
print("  The Adjusted Rand Index (ARI) corrects RI for chance agreement.")
print("  It uses the same binary contingency (n11, n10, n01) as RI.")
print()
print("  Step 1: Compute marginals of the 2x2 contingency table:")
print("            row margins:  a0 = n01          a1 = n10 + n11")
print("            col margins:  b0 = n10          b1 = n01 + n11")
print("  Step 2: Compute pair-count sums:")
print("            T_cells = C(n11,2) + C(n10,2) + C(n01,2)")
print("            T_rows  = C(a0,2)  + C(a1,2)")
print("            T_cols  = C(b0,2)  + C(b1,2)")
print("  Step 3: Expected index (under random permutations):")
print("            E = (T_rows * T_cols) / C(n, 2)")
print("  Step 4: Maximum index:")
print("            M = 0.5 * (T_rows + T_cols)")
print("  Step 5: ARI = (T_cells - E) / (M - E).")
print("          When M = E (degenerate case), ARI = 1.")

_section("Input")
print("  Same clusters: A0={0,1,2}, A1={3,4}, B0={0,1}, B1={2,3,4}")
print()
print("  Pair (A0, B0): n11=2, n10=1, n01=0, n=3, C(3,2)=3")
print("    T_cells = C(2,2)+C(1,2)+C(0,2) = 1+0+0 = 1")
print("    a0=0, a1=3 -> T_rows = C(0,2)+C(3,2) = 0+3 = 3")
print("    b0=1, b1=2 -> T_cols = C(1,2)+C(2,2) = 0+1 = 1")
print("    E = (3*1)/3 = 1   M = 0.5*(3+1) = 2")
print("    ARI = (1-1)/(2-1) = 0/1 = 0.000")
print()
print("  Pair (A0, B1): n11=1, n10=2, n01=2, n=5, C(5,2)=10")
print("    T_cells = C(1,2)+C(2,2)+C(2,2) = 0+1+1 = 2")
print("    a0=2, a1=3 -> T_rows = C(2,2)+C(3,2) = 1+3 = 4")
print("    b0=2, b1=3 -> T_cols = C(2,2)+C(3,2) = 1+3 = 4")
print("    E = (4*4)/10 = 1.6   M = 0.5*(4+4) = 4")
print("    ARI = (2-1.6)/(4-1.6) = 0.4/2.4 = 1/6 ~ 0.167")
print()
print("  Pair (A1, B0): n11=0, n10=2, n01=2, n=4, C(4,2)=6")
print("    T_cells = C(0,2)+C(2,2)+C(2,2) = 0+1+1 = 2")
print("    a0=2, a1=2 -> T_rows = C(2,2)+C(2,2) = 1+1 = 2")
print("    b0=2, b1=2 -> T_cols = C(2,2)+C(2,2) = 1+1 = 2")
print("    E = (2*2)/6 = 2/3   M = 0.5*(2+2) = 2")
print("    ARI = (2-2/3)/(2-2/3) = 1.000  [numerator equals denominator]")
print()
print("  Pair (A1, B1): n11=2, n10=0, n01=1, n=3, C(3,2)=3")
print("    T_cells = C(2,2)+C(0,2)+C(1,2) = 1+0+0 = 1")
print("    a0=1, a1=2 -> T_rows = C(1,2)+C(2,2) = 0+1 = 1")
print("    b0=0, b1=3 -> T_cols = C(0,2)+C(3,2) = 0+3 = 3")
print("    E = (1*3)/3 = 1   M = 0.5*(1+3) = 2")
print("    ARI = (1-1)/(2-1) = 0/1 = 0.000")

ARI_clust = RV.adjusted_rand_index_clusters(Sol1_clusters, Sol2_clusters)

expected_ARI_clust = np.array([
    [0.0,   1/6],
    [1.0,   0.0],
])

_section("Output  [2x2 cluster ARI matrix]")
_result(ARI_clust, expected_ARI_clust)
