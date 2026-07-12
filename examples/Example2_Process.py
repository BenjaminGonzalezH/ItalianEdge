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
1. consensus_matrix                 (consensus_matrix.py)
2. jaccard_index_solutions          (jaccard_values.py)
3. jaccard_index_clusters           (jaccard_values.py)
4. rand_index_clusters              (rand_values.py -- _rand_from_binary_contingency)
5. adjusted_rand_index_clusters     (rand_values.py -- _ari_from_binary_contingency)
6. _align_partition                 (plurality_voting.py)
7. compute_hierarchical_clustering  (he_clustering.py)
8. compute_inconsistency_clustering (he_inconsistency_clustering.py)
"""

import numpy                                                             as np
import gclusters_characterization.clustering.consensus_matrix            as CM
import gclusters_characterization.clustering.jaccard_values              as JV
import gclusters_characterization.clustering.rand_values                 as RV
import gclusters_characterization.clustering.he_clustering               as HC
import gclusters_characterization.clustering.he_inconsistency_clustering as HIC


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


# ======================================================
# 6. compute_hierarchical_clustering
# ======================================================

_header("compute_hierarchical_clustering  [he_clustering.py]")

_section("Theory")
print("  Hierarchical clustering repeatedly merges the two closest")
print("  clusters (single linkage: the distance between two clusters")
print("  is the MINIMUM distance between any of their members) until")
print("  a single cluster remains, recording each merge's height.")
print()
print("  Step 1: Start with every element as its own cluster.")
print("  Step 2: Merge the two closest clusters; record the merge")
print("          height (= their distance) in the linkage matrix Z.")
print("  Step 3: Repeat step 2, using single-linkage distance between")
print("          the newly formed cluster and every remaining one,")
print("          until only one cluster is left (n-1 merges total).")
print("  Step 4: fcluster(Z, num_groups, criterion='maxclust') cuts")
print("          the tree to return exactly num_groups flat labels.")
print("  Step 5: The cophenetic distance between two leaves is the")
print("          height at which they first end up in the same")
print("          cluster. The cophenetic correlation coefficient is")
print("          the Pearson correlation between the original pairwise")
print("          distances and these cophenetic distances -- it")
print("          measures how faithfully the dendrogram preserves the")
print("          original distance matrix (1.0 = perfect preservation).")

_section("Input")
print("  4 genes (0-3), precomputed distance matrix, single linkage,")
print("  num_groups=2:")
print()
print("    D = [[0, 1, 5, 6],")
print("         [1, 0, 5, 6],")
print("         [5, 5, 0, 2],")
print("         [6, 6, 2, 0]]")
print()
print("  Step 2 (1st merge): smallest distance overall is d(G0,G1)=1")
print("    -> merge {G0,G1} at height 1.")
print("  Step 2 (2nd merge): smallest remaining distance is d(G2,G3)=2")
print("    -> merge {G2,G3} at height 2.")
print("  Step 2 (3rd merge): single-linkage distance between {G0,G1}")
print("    and {G2,G3} = min(d(G0,G2),d(G0,G3),d(G1,G2),d(G1,G3))")
print("               = min(5,6,5,6) = 5")
print("    -> merge {G0,G1,G2,G3} at height 5.")
print()
print("  Cutting at num_groups=2 keeps the first two merges intact and")
print("  undoes only the last one -> labels: G0,G1 -> cluster A;")
print("  G2,G3 -> cluster B.")
print()
print("  Cophenetic distances (height of first shared cluster):")
print("    coph(G0,G1)=1  coph(G2,G3)=2  all cross pairs (G0/G1 vs")
print("    G2/G3) = 5 (the final merge height).")
print()
print("  orig = [d(0,1), d(0,2), d(0,3), d(1,2), d(1,3), d(2,3)]")
print("       = [1, 5, 6, 5, 6, 2]           mean = 25/6 ~ 4.1667")
print("  coph = [1, 5, 5, 5, 5, 2]           mean = 23/6 ~ 3.8333")
print("  Pearson correlation of (orig, coph) ~ 0.9776")

D = np.array([
    [0., 1., 5., 6.],
    [1., 0., 5., 6.],
    [5., 5., 0., 2.],
    [6., 6., 2., 0.],
])
genes = ["G0", "G1", "G2", "G3"]

hc_options = HC.ClusteringOptions(num_groups=2, method="single", validate_distance=True, verbose=False)
Z, labels, cophenetic_corr = HC.compute_hierarchical_clustering(D, genes, hc_options)

expected_Z = np.array([
    [0., 1., 1., 2.],
    [2., 3., 2., 2.],
    [4., 5., 5., 4.],
])
expected_labels = np.array([1, 1, 2, 2])
expected_cophenetic_corr = 0.9776352896442044

_section("Output  [linkage matrix Z: (leaf/cluster 1, leaf/cluster 2, height, size)]")
_result(Z, expected_Z)

_section("Output  [flat cluster labels, num_groups=2]")
_result(labels, expected_labels)

_section("Output  [cophenetic correlation coefficient]")
_result(cophenetic_corr, expected_cophenetic_corr)


# ======================================================
# 8. compute_inconsistency_clustering
# ======================================================

_header("compute_inconsistency_clustering  [he_inconsistency_clustering.py]")

_section("Theory")
print("  Instead of asking the user for num_groups, this flags the merge")
print("  whose height is most anomalous RELATIVE TO ITS OWN LOCAL")
print("  NEIGHBOURHOOD in the dendrogram, then cuts just below it.")
print()
print("  Step 1: Build the same linkage matrix Z as compute_hierarchical_")
print("          clustering (single linkage, see section 7).")
print("  Step 2: For every merge/link, gather the heights of that link")
print("          and every link within `depth` levels below it in the")
print("          tree (depth=2 by default). Compute the sample mean and")
print("          sample standard deviation (ddof=1) of that height set.")
print("  Step 3: inconsistency coefficient = (own height - mean) / std")
print("          (0 when std == 0, e.g. a merge with no sub-links).")
print("  Step 4: Rank merges by coefficient, descending. The candidate's")
print("          k = n_leaves - merge_index (clusters if that merge is")
print("          the one left undone), filtered to min_clusters <= k")
print("          <= max_clusters. cut_height = midpoint between this")
print("          merge's height and the previous one (0 for the first).")
print("  Step 5: fcluster(Z, t=cut_height, criterion='distance') turns")
print("          the winning cut_height into flat labels.")

_section("Input")
print("  Same 4 genes and distance matrix D as section 7, single")
print("  linkage, default options (depth=2, n_candidates=1,")
print("  min_clusters=2). Reuses the same Z:")
print()
print("    Z = [[0, 1, 1, 2],   <- merge0: {G0,G1} at height 1")
print("         [2, 3, 2, 2],   <- merge1: {G2,G3} at height 2")
print("         [4, 5, 5, 4]]   <- merge2: {G0,G1,G2,G3} at height 5")
print()
print("  merge0 and merge1 only join two leaves -> no sub-links below")
print("  them -> neighbourhood = {their own height}, std=0")
print("    -> coefficient(merge0) = 0,  coefficient(merge1) = 0.")
print()
print("  merge2's neighbourhood (itself + both children, depth=2):")
print("    heights = {5 (self), 1 (merge0), 2 (merge1)}")
print("    mean = (5+1+2)/3 = 8/3 ~ 2.6667")
print("    sample variance = [(5-2.6667)^2+(1-2.6667)^2+(2-2.6667)^2] / (3-1)")
print("                     = [5.444+2.778+0.444] / 2 = 8.667/2 = 4.333")
print("    std = sqrt(4.333) ~ 2.0817")
print("    coefficient(merge2) = (5 - 2.6667) / 2.0817 = 2.3333/2.0817 ~ 1.1209")
print()
print("  merge2 is the clear outlier (coefficient ~1.12 vs 0 for the")
print("  other two) -> it is flagged as the natural cut point.")
print()
print("  Candidate: undo merge2 -> k = n_leaves - 2 = 4 - 2 = 2 clusters.")
print("  cut_height = midpoint(height[merge1], height[merge2])")
print("             = (2 + 5) / 2 = 3.5")
print("  Cutting Z at height 3.5 keeps merge0/merge1 (heights 1, 2) but")
print("  not merge2 (height 5) -> same 2 clusters as section 7:")
print("  {G0,G1} and {G2,G3}.")

ic_options = HIC.InconsistencyClusteringOptions(
    method="single", depth=2, n_candidates=1, min_clusters=2, validate_distance=True, verbose=False
)
Z_ic, labels_ic, cophenetic_corr_ic, report, R = HIC.compute_inconsistency_clustering(D, genes, ic_options)
best = report[0]

expected_R_coefficients = np.array([0.0, 0.0, 1.12089707663561])
expected_best_k = 2
expected_best_cut_height = 3.5
expected_labels_ic = np.array([1, 1, 2, 2])

_section("Output  [inconsistency coefficient per merge, R[:,3]]")
_result(R[:, 3], expected_R_coefficients)

_section("Output  [best candidate: number of clusters k]")
_result(best["k"], expected_best_k)

_section("Output  [best candidate: cut height]")
_result(best["cut_height"], expected_best_cut_height)

_section("Output  [flat cluster labels from the auto-detected cut]")
_result(labels_ic, expected_labels_ic)
