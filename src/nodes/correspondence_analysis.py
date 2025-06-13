import logging
import knime.extension as knext
import pandas as pd
import numpy as np
from util import utils as kutil
import matplotlib.pyplot as plt
from io import BytesIO

LOGGER = logging.getLogger(__name__)

@knext.node(
    name="Correspondence Analysis",
    node_type=knext.NodeType.LEARNER,
    icon_path="icons/models/correspondence.png",  # Replace with relevant icon
    category=kutil.category_dimensionality_reduction,
    id="correspondence_analysis",
)
@knext.input_table(
    name="Input Data",
    description="Table containing two or more categorical columns for Correspondence Analysis or MCA.",
)
@knext.output_table(
    name="Variance Explained",
    description="Percentage of variance explained by the requested dimensions.",
)
@knext.output_table(
    name="Model Summary",
    description="Contributions of each modality to the principal dimensions, coordinates in the new space, quality of the representation.",
)

@knext.output_image(
    name="Factors Map",
    description="It shows the factors map of the Correspondence Analysis.",
)

class CorrespondenceAnalysisNode:
    """
    A KNIME learner node that performs Correspondence Analysis (CA) or Multiple Correspondence Analysis (MCA)  on one or more categorical columns.

    **Model Overview:**
    This node is designed to extract latent associations between categorical variables by projecting them into a lower-dimensional Euclidean space. It supports both:

    - **CA**: for exactly two categorical columns (via a contingency table)

    - **MCA**: for more than two categorical columns (via a complete disjunctive table)

    The node computes principal coordinates, eigenvalues, contributions, and cosine squared values (cos²), providing insight into the structure of associations between modalities. It also generates a factor map to visualize the relationships among categories in the selected dimensions.

    **Inputs:**

    - A KNIME table containing two or more categorical columns (string type).

    **Parameters:**

    - `Categorical Input Columns`: The string columns used for the analysis.

    - `Number of Output Dimensions`: The number of dimensions to compute and visualize (typically 2).

    **Outputs:**

    - **Variance Explained Table**: Contains the eigenvalues, proportion of variance explained by each dimension, and cumulative variance.

    - **Model Summary Table**: Reports statistics for each modality (category) including mass, contribution, coordinates, and representation quality (cos²).

    - **Factor Map Image**: A 2D plot showing modality positions and groupings in the selected factor space.

    **Model Summary Explanation (per modality):**

    - **Modality**:  
      The name of the category (e.g., `"Email"`, `"TV"`, `"High"`) being analyzed. Extracted from the selected input columns.

    - **Mass**:  
      The relative frequency (marginal proportion) of the modality in the dataset. Higher mass values indicate more common categories.

    - **Point Inertia**:  
      The absolute contribution of the modality to the total inertia (i.e., variance), calculated as `mass multiplied by the squared distance from origin`.

    - **Contribution (Dim #)**:  
      The percentage contribution of the modality to the inertia of each dimension. Higher values mean the modality plays a stronger role in shaping that axis.

    - **Coordinate (Dim #)**:  
      The position of the modality along each principal dimension. These values are used to place points on the factor map. Categories that appear close together on the map are more similar.

    - **cos² (Dim #)** *(aka Representation Quality)*:  
      The squared cosine of the angle between the modality vector and the dimension axis. Indicates how well the modality is represented by that dimension. Values near 1 mean excellent representation. If you sum the cos² values across the extracted dimensions, you get the overall representation quality for that modality.

    **Error Handling:**
    - Ensures at least two selected columns have at least two distinct values each.

    - Filters out near-zero eigenvalues to avoid numerical instability.

    - Automatically reduces the number of computed dimensions based on the rank of the data.

    - Applies Benzécri correction to eigenvalues in MCA mode for interpretability.

    This node is useful for exploratory data analysis in surveys, market segmentation, and identifying hidden patterns in categorical datasets.
    """

    features_cols = knext.MultiColumnParameter(
        label="Categorical Input Columns",
        description="Select one or more categorical columns to include in the correspondence analysis",
        column_filter=kutil.is_string,
    )

    n_components = knext.IntParameter(
        label="Number of Output Dimensions",
        description="Specify how many principal dimensions (axes) to compute for the analysis. This determines how much of the total inertia (variance) is retained in the lower-dimensional output.",
        default_value=2,
        min_value=1,
        max_value=100,
    )

    def configure(self, configure_context: knext.ConfigurationContext, input_schema: knext.Schema):
        num_cols = len(self.features_cols)
        max_dims = min(self.n_components, num_cols)  # enforce upper limit

        variance_explained_schema = knext.Schema(
            [knext.double(), knext.double(), knext.double()],
            ["Eigenvalue", "Explained Variance Ratio","Cumulative Explained Variance"]
        )

        contrib_score_schema = knext.Schema(
            [knext.string(), knext.double(), knext.double()] + [knext.double()] * (3 * max_dims),
            ["Modality", "Mass", "Point Inertia"] +
            [f"Contribution (Dim {i+1})" for i in range(max_dims)] +
            [f"Coordinate (Dim {i+1})" for i in range(max_dims)] +
            [f"cos² (Dim {i+1})" for i in range(max_dims)]
)

        return (variance_explained_schema, 
                contrib_score_schema, 
                knext.ImagePortObjectSpec(knext.ImageFormat.SVG),
        )
    
    def execute(self, exec_context: knext.ExecutionContext, input_table: knext.Table):
        df = input_table.to_pandas()
        dimensions = self.features_cols
        dimension_df = df[dimensions].fillna("missing").astype(str)
        max_dims = self.n_components

        if df.size == 0:
            raise ValueError("Input table is empty. Please provide data to analyze.")

        if len(dimensions) < 2:
            raise ValueError("Please select at least two categorical columns for analysis.")
      
        for col in dimensions:
            if dimension_df[col].nunique() < 2:
                raise ValueError(f"Column '{col}' must have at least 2 unique values.")

        if len(dimensions) == 2:
            # Step 1: Compute contingency table
            contingency = pd.crosstab(dimension_df[dimensions[0]], dimension_df[dimensions[1]])
            observed = contingency.to_numpy()

            # === Step 2: Normalize counts to get correspondence matrix ===
            X = observed.astype(float) / observed.sum()  # equivalent to prince

            # === Step 3: Compute marginal distributions (masses) ===
            r = X.sum(axis=1)  # row masses
            c = X.sum(axis=0)  # column masses

            # === Step 4: Compute standardized residuals matrix ===
            S = np.diag(1.0 / np.sqrt(r)) @ (X - np.outer(r, c)) @ np.diag(1.0 / np.sqrt(c))

            # === Step 5: Apply SVD ===
            max_dims = min(self.n_components, min(observed.shape) - 1)
            U, singular_vals, VT = np.linalg.svd(S, full_matrices=False)

            # === Step 6: Filter eigenvalues close to 0, computing the variance explained and the total inertia ===
            all_eigenvals = singular_vals**2
            threshold = 1e-12
            valid = all_eigenvals > threshold
            eigenvals = all_eigenvals[valid]
            singular_vals = singular_vals[valid]
            U = U[:, valid]
            VT = VT[valid, :]

            if len(eigenvals) < self.n_components:
                raise ValueError(
                    f"Only {len(eigenvals)} components could be computed after eigenvalue filtering. Eingevalues close to zero are filtered out. "
                    f"Requested {self.n_components}, but data rank is lower. It means there is low variance in your data. Try fewer dimensions or include more categorical diversity."
                )

            total_inertia = np.sum(all_eigenvals)
            explained_ratio = eigenvals / total_inertia

            # === Step 7: Compute coordinates and representation quality with respect to the dimensions ===
            row_coords = np.diag(1.0 / np.sqrt(r)) @ U * singular_vals
            col_coords = np.diag(1.0 / np.sqrt(c)) @ VT.T * singular_vals
            
            cos2_row = (row_coords ** 2) / np.sum(row_coords ** 2, axis=1, keepdims=True)
            cos2_col = (col_coords ** 2) / np.sum(col_coords ** 2, axis=1, keepdims=True)

            # === Step 8: Contributions to dimensions and rows and columns masses ===
            row_contrib = (row_coords**2) / eigenvals
            col_contrib = (col_coords**2) / eigenvals

            row_labels = contingency.index.astype(str)
            col_labels = contingency.columns.astype(str)
            masses = np.concatenate([r, c])

            # === Step 9: Combine results ===
            scores_matrix = np.vstack([row_coords, col_coords])
            modality_labels = list(row_labels) + list(col_labels)
            contrib_matrix = np.vstack([row_contrib, col_contrib])
            cos_matrix = col_cos2 = np.vstack([cos2_row, cos2_col])

        else:
            # Step 1: Create indicator matrix using pandas
            Z = pd.get_dummies(dimension_df, columns=dimension_df.columns)

            if Z.shape[1] == 0:
                raise ValueError("MCA failed: no valid one-hot encoded features found.")

            # Optional tracking of dimensions
            K = len(dimensions)  # number of original categorical variables
            J = Z.shape[1]        # total number of dummies (indicator columns)

            # Step 2: Normalize to correspondence matrix
            Z = Z / Z.values.sum()

            # Step 3: Row and column masses
            r = Z.sum(axis=1).to_numpy()
            c = Z.sum(axis=0).to_numpy()

            # Step 4: Standardized residual matrix
            S = np.diag(1.0 / np.sqrt(r)) @ (Z.to_numpy() - np.outer(r, c)) @ np.diag(1.0 / np.sqrt(c))

            # Step 5: SVD
            U, singular_vals, VT = np.linalg.svd(S, full_matrices=False)
            all_eigenvals = singular_vals**2

            # Step 6: Filter small eigenvalues and slice the matrices
            threshold = 1e-12
            valid = all_eigenvals > threshold
            eigenvals = all_eigenvals[valid]
            singular_vals = singular_vals[valid]
            U = U[:, valid]
            VT = VT[valid, :]

            if len(eigenvals) < self.n_components:
                raise ValueError(
                    f"Only {len(eigenvals)} components could be computed after eigenvalue filtering. "
                    f"Requested {self.n_components}, but data rank is lower. Try reducing dimensions or improving categorical variance."
                )

            #  Benzécri correction to make MCA more interpretable and similar to PCA
            if K > 1:
                corrected_eigenvals = np.array([
                    (K / (K - 1) * (eig - 1 / K))**2 if eig > 1 / K else 0 for eig in eigenvals
                ])
                eigenvals = corrected_eigenvals

            # Step 7: Coordinates, contributions, representation quality respective to dimensions
            row_coords = np.diag(1.0 / np.sqrt(r)) @ U * singular_vals
            col_coords = np.diag(1.0 / np.sqrt(c)) @ VT.T * singular_vals
            
            row_contrib = (row_coords**2) / eigenvals
            col_contrib = (col_coords**2) / eigenvals

            cos2_col = (col_coords ** 2) / np.sum(col_coords ** 2, axis=1, keepdims=True)
            col_cos2 = cos2_col  # MCA uses only column modalities

            # Step 8: Combine outputs
            modality_labels = list(Z.columns.astype(str))
            scores_matrix = col_coords
            contrib_matrix = col_contrib
            cos_matrix=col_cos2

            # MCA uses only column categories
            masses = c  

            # Total inertia and explained ratio
            total_inertia = np.sum(all_eigenvals)
            explained_ratio = eigenvals / total_inertia

       # === Create  factor map ===
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.axhline(0, color='gray', lw=1)
        ax.axvline(0, color='gray', lw=1)
        ax.grid(True, linestyle='--', alpha=0.5)

        if scores_matrix.shape[1] < 2:
            raise ValueError("Not enough components to plot factor map. Need at least 2 dimensions.")

        x = scores_matrix[:, 0]
        y = scores_matrix[:, 1]

        # === Axis bounds with fixed ±0.2 padding, no minimum range enforcement
        x_margin = 0.2
        y_margin = 0.2
        xmin, xmax = np.min(x) - x_margin, np.max(x) + x_margin
        ymin, ymax = np.min(y) - y_margin, np.max(y) + y_margin

        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)

        # === Assign styles by column ===
        if len(dimensions) == 2:
            modality_columns = [dimensions[0]] * len(row_labels) + [dimensions[1]] * len(col_labels)
        else:
            modality_columns = [col.split('_')[0] for col in modality_labels]

        column_names = sorted(set(modality_columns))
        base_colors = plt.get_cmap('tab10').colors
        base_markers = ['o', 's', '^', 'D', 'v', 'P', '*', 'X']

        column_style_map = {
            col: {"color": base_colors[i % len(base_colors)],
                  "marker": base_markers[i % len(base_markers)]}
            for i, col in enumerate(column_names)
        }

        # === Plot points and prepare legend handles ===
        legend_handles = {}
        for i, label in enumerate(modality_labels):
            col = modality_columns[i]
            style = column_style_map[col]
            short_label = label.split('_')[-1] if '_' in label else label

            ax.scatter(x[i], y[i],
                       color=style["color"],
                       marker=style["marker"],
                       edgecolor='black',
                       s=70, alpha=0.9, zorder=3)

            if len(column_names) <= 4 and col not in legend_handles:
                legend_handles[col] = ax.scatter([], [], color=style["color"],
                                                 marker=style["marker"], label=col)


        # === Smart label placement with adaptive offset after first cycle ===
        base_offset = 0.025 * max(np.ptp(x), np.ptp(y))
        max_cycles = 3  # 1st cycle fixed offset, next ones increase radius

        # Unit direction vectors (no embedded offset)
        directions = [(1, 1), (-1, 1), (-1, -1), (1, -1),
                    (-1, 0), (1, 0), (0, 1), (0, -1)]

        label_positions = []

        def is_too_close(p1, p2, threshold=0.04):
            return np.hypot(p1[0] - p2[0], p1[1] - p2[1]) < threshold

        for i, label in enumerate(modality_labels):
            xi, yi = x[i], y[i]
            col = modality_columns[i]
            short_label = label.split('_')[-1] if '_' in label else label

            # Try increasing offset after first full cycle
            for cycle_i in range(max_cycles):
                scale = 1.0 if cycle_i == 0 else 1.5 * cycle_i  # Adaptive growth after first round
                for dx, dy in directions:
                    candidate_pos = (
                        xi + dx * base_offset * scale,
                        yi + dy * base_offset * scale
                    )
                    if not any(is_too_close(candidate_pos, pos) for pos in label_positions):
                        break  # Found a good position
                else:
                    continue  # Try next cycle
                break  # Success: break outer loop
            else:
                LOGGER.warning(f"Label '{label}' placed at fallback due to crowding.")

            label_positions.append(candidate_pos)
            ax.text(candidate_pos[0], candidate_pos[1], short_label,
                    fontsize=9, ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1.5))


        # === Axis labels and final layout ===
        ax.set_xlabel("Dimension 1", fontsize=12)
        ax.set_ylabel("Dimension 2", fontsize=12)
        ax.set_title("Correspondence Analysis – Factor Map", fontsize=14, weight='bold')
        ax.set_aspect('equal')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        if len(column_names) <= 4:
            ax.legend(handles=list(legend_handles.values()), loc='best', fontsize=9, title="Variable")

        plt.tight_layout()
        plt.savefig("factor_map.svg", dpi=300, bbox_inches='tight')

        # Save as SVG to in-memory buffer
        buf = BytesIO()
        fig.savefig(buf, format='svg')
        
        # Variance explained output
        result_df = pd.DataFrame({
            "Eigenvalue": eigenvals[:max_dims],
            "Explained Variance Ratio": explained_ratio[:max_dims],
            "Cumulative Explained Variance": explained_ratio[:max_dims].cumsum()
        })

        # Pad contributions matrix if needed
        if contrib_matrix.shape[1] < max_dims:
            pad_width = max_dims - contrib_matrix.shape[1]
            contrib_matrix = np.hstack([contrib_matrix, np.zeros((contrib_matrix.shape[0], pad_width))])

        # Pad scores matrix if needed
        if scores_matrix.shape[1] < max_dims:
            pad_width = max_dims - scores_matrix.shape[1]
            scores_matrix = np.hstack([scores_matrix, np.zeros((scores_matrix.shape[0], pad_width))])

        # Pad scores matrix if needed
        if cos_matrix.shape[1] < max_dims:
            pad_width = max_dims - cos_matrix.shape[1]
            cos_matrix = np.hstack([cos_matrix, np.zeros((cos_matrix.shape[0], pad_width))])

        # Compute Point Inertia: mass * squared Euclidean norm across components
        scores_used = scores_matrix[:, :max_dims]
        squared_norms = np.sum(scores_used**2, axis=1)
        point_inertia = masses * squared_norms

        # Now safely build DataFrames
        contrib_df = pd.DataFrame(
            contrib_matrix[:, :max_dims],
            columns=[f"Contribution (Dim {i+1})" for i in range(max_dims)]
        )
        contrib_df.insert(0, "Modality", modality_labels)

        score_df = pd.DataFrame(
            scores_matrix[:, :max_dims],
            columns=[f"Coordinate (Dim {i+1})" for i in range(max_dims)]
        )
        cos2_df = pd.DataFrame(
            cos_matrix[:, :max_dims],
            columns=[f"cos² (Dim {i+1})" for i in range(max_dims)]
        )

        # Combine DataFrames
        contrib_score_df = pd.concat([contrib_df.reset_index(drop=True), score_df, cos2_df], axis=1)

        # Insert Mass and Point Inertia right after Modality
        contrib_score_df.insert(1, "Mass", masses)
        contrib_score_df.insert(2, "Point Inertia", point_inertia)

        return (
            knext.Table.from_pandas(result_df),
            knext.Table.from_pandas(contrib_score_df),
            buf.getvalue(),
        )