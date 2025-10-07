import argparse
import pandas as pd
import scanpy as sc
from immunopheno.data_processing import ImmunoPhenoData

import umap
import seaborn as sns
import matplotlib.pyplot as plt

def plot_UMAP(IPD: ImmunoPhenoData,
              normalized: bool = False,
              force_update: bool = False,
              random_state: int = 42,
              **kwargs):
    """ Plots a UMAP for the non-normalized protein values or normalized protein values

    Args:
        IPD (ImmunoPhenoData Object): object containing protein data,
            gene data, and cell types
        normalized (bool): option to plot normalized values
        force_update (bool): option to compute a new UMAP, replacing stored plots
        **kwargs: various arguments to UMAP class constructor, including default values:
            n_neighbors (int): 15
            min_dist (float): 0.1 
            n_components (int): 2
            metric (str): "euclidean"

    Returns:
        go.Figure: UMAP projection of non/normalized protein values with a corresponding
        legend of cell type (if available)
    """
    # Check if existing UMAP is present in class AND UMAP parameters have not changed
    if (IPD._umap_kwargs == (random_state, kwargs) and IPD._raw_umap is not None) and normalized is False and force_update is False:
        # If so, return the stored UMAP
        return IPD._raw_umap
    elif (IPD._umap_kwargs == (random_state, kwargs) and IPD._norm_umap is not None) and normalized is True and force_update is False:
        return IPD._norm_umap
    else:
        # If no UMAP or kwargs are different, generate a new one and store in class
        umap_plot = umap.UMAP(random_state=random_state, **kwargs)

        # Store new kwargs in class
        IPD._umap_kwargs = (random_state, kwargs)
        
        if normalized:
            norm_projections = umap_plot.fit_transform(IPD.normalized_counts)
        else:
            raw_projections = umap_plot.fit_transform(IPD.protein)
        
        # Normalized UMAP without cell labels
        if IPD.labels is None and normalized:
            # Create Seaborn plot
            fig, ax = plt.subplots(figsize=(16, 6))
            sns.scatterplot(
                x=norm_projections[:, 0],
                y=norm_projections[:, 1],
                s=7,
                ax=ax,
            )
            ax.set_title("Normalized Protein Expression UMAP")
            plt.tight_layout()
            return fig

        # Un-normalized UMAP without cell labels
        elif IPD._cell_labels is None and not normalized:
            # Create Seaborn plot
            fig, ax = plt.subplots(figsize=(16, 6))
            sns.scatterplot(
                x=raw_projections[:, 0],
                y=raw_projections[:, 1],
                s=7,
                ax=ax,
            )
            ax.set_title("Regular Protein Expression UMAP")
            plt.tight_layout()
            return fig

        # Normalized UMAP plot with cell labels
        if IPD.labels is not None and normalized:
            # NOTE: if the provided labels contains more cells than present in normalized_counts
            # Find shared index in the IPD.labels based on cells ONLY in normalized_counts
            common_indices = IPD.normalized_counts.index.intersection(IPD.labels.index)

            # Check the number of columns
            num_columns = IPD.labels.shape[1]
            # Check if there is at least one column and if the second column is not empty
            if num_columns > 1 and not IPD.labels.iloc[:, 1].isnull().all():
                # Use the values from the second column
                norm_types = IPD.labels.iloc[:, 1].loc[common_indices].tolist()

            else:
                # If there is no second column or it is empty, use the values from the first column
                norm_types = IPD.labels.iloc[:, 0].loc[common_indices].tolist()

            # Create Seaborn plot
            fig, ax = plt.subplots(figsize=(16, 6))
            sns.scatterplot(
                x=norm_projections[:, 0],
                y=norm_projections[:, 1],
                hue=norm_types,
                palette="tab20",
                s=7,
                ax=ax,
                legend="full"
            )
            ax.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
            ax.set_title("Normalized Protein Expression UMAP")
            plt.tight_layout()
            return fig
            
        # Not normalized UMAP plot with cell labels
        elif IPD._cell_labels is not None and not normalized:
            # Check the number of columns
            num_columns = IPD._cell_labels.shape[1]
            # Check if there is at least one column and if the second column is not empty
            if num_columns > 1 and not IPD._cell_labels.iloc[:, 1].isnull().all():
                # Use the values from the second column
                raw_types = IPD._cell_labels.iloc[:, 1].tolist()
            else:
                # If there is no second column or it is empty, use the values from the first column
                raw_types = IPD._cell_labels.iloc[:, 0].tolist()

            # Create Seaborn plot
            fig, ax = plt.subplots(figsize=(16, 6))
            sns.scatterplot(
                x=raw_projections[:, 0],
                y=raw_projections[:, 1],
                hue=raw_types,
                palette="tab20",
                s=7,
                ax=ax,
                legend="full"
            )
            ax.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
            ax.set_title("Regular Protein Expression UMAP")
            plt.tight_layout()
            return fig
        
def combine_umap_plots(IPD, **kwargs):
    """
    Generates two UMAP plots (normalized and unnormalized) and combines them into a single figure.

    Args:
        IPD: ImmunoPhenoData object containing protein and label data.
        **kwargs: Optional arguments to pass to the UMAP class constructor.

    Returns:
        None: Saves the combined figure as a PNG.
    """
    # Call your existing plot_UMAP function twice to generate two figures
    fig_raw = plot_UMAP(IPD=IPD, normalized=False, **kwargs)   # Unnormalized plot
    fig_norm = plot_UMAP(IPD=IPD, normalized=True, **kwargs)   # Normalized plot

    # Extract the axes from the two existing figures
    ax_raw = fig_raw.axes[0]
    ax_norm = fig_norm.axes[0]

    # Create a new figure with two vertically stacked subplots
    combined_fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))  # Two rows, one column

    # Copy data from the original axes to the new subplots
    # Copying for the unnormalized plot (top)
    for line in ax_raw.lines:
        ax1.plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), 
                 color=line.get_color(), marker='o', markersize=1)
    for col in ax_raw.collections:
        ax1.scatter(col.get_offsets()[:, 0], col.get_offsets()[:, 1], 
                    color=col.get_facecolor(), s=7)

    # Copying for the normalized plot (bottom)
    for line in ax_norm.lines:
        ax2.plot(line.get_xdata(), line.get_ydata(), linestyle=line.get_linestyle(), 
                 color=line.get_color(), marker='o', markersize=1)
    for col in ax_norm.collections:
        ax2.scatter(col.get_offsets()[:, 0], col.get_offsets()[:, 1], 
                    color=col.get_facecolor(), s=7)

    # Set titles for the subplots
    ax1.set_title("Regular Protein Expression UMAP")
    ax2.set_title("Normalized Protein Expression UMAP")

    # Add legends outside of the axes
    if ax_raw.get_legend() is not None:
        ax1.legend(*ax_raw.get_legend_handles_labels(), title="Cell Type", 
                   loc='upper left', bbox_to_anchor=(1.05, 1), markerscale=2)  # Right of plot
    if ax_norm.get_legend() is not None:
        ax2.legend(*ax_norm.get_legend_handles_labels(), title="Cell Type", 
                   loc='upper left', bbox_to_anchor=(1.05, 1), markerscale=2)  # Right of plot

    # Adjust layout to fit the legends
    plt.subplots_adjust(right=0.8, hspace=0.4)  # Give extra space for legends and adjust height

    # Adjust layout and save the combined figure
    plt.tight_layout()
    return combined_fig

def main(model, 
         output_csv,
         output_umap, 
         protein_csv_file_path=None,
         rna_csv_file_path=None,
         labels_csv_file_path=None,
         scanpy_path=None, 
         scanpy_labels=None,
         sig_expr_threshold=1, 
         bg_expr_threshold=0, 
         p_threshold=0.05, 
         bg_cell_z_score=10.0):
    
    """
    Normalize CITE-Seq data (protein) and produce a UMAP
    """
    
    # CITE-Seq data 
    if protein_csv_file_path and not scanpy_path:
        # Protein only
        if not rna_csv_file_path and not labels_csv_file_path:
            ipd = ImmunoPhenoData(
                protein_matrix=protein_csv_file_path,
            )
            ipd.fit_all_antibodies(model=model)
            ipd.normalize_all_antibodies(sig_expr_threshold=sig_expr_threshold, 
                                        bg_expr_threshold=bg_expr_threshold, 
                                        p_threshold=p_threshold, 
                                        bg_cell_z_score=bg_cell_z_score)
            ipd.normalized_counts.to_csv(output_csv)
            print(f"Normalized counts saved to {output_csv}")

            print("Generating UMAPs...")
            double_UMAP = combine_umap_plots(IPD=ipd).savefig(output_umap)
            print("UMAP saved to:", output_umap)

        # Protein and RNA
        elif rna_csv_file_path and not labels_csv_file_path:
            ipd = ImmunoPhenoData(
                protein_matrix=protein_csv_file_path,
                gene_matrix=rna_csv_file_path,
            )
            ipd.fit_all_antibodies(model=model)
            ipd.normalize_all_antibodies(sig_expr_threshold=sig_expr_threshold, 
                                        bg_expr_threshold=bg_expr_threshold, 
                                        p_threshold=p_threshold, 
                                        bg_cell_z_score=bg_cell_z_score)
            ipd.normalized_counts.to_csv(output_csv)
            print(f"Normalized counts saved to {output_csv}")

            print("Generating UMAPs...")
            double_UMAP = combine_umap_plots(IPD=ipd).savefig(output_umap)
            print("UMAP saved to:", output_umap)

        # Protein, RNA, labels
        elif rna_csv_file_path and labels_csv_file_path:
            ipd = ImmunoPhenoData(
                protein_matrix=protein_csv_file_path,
                gene_matrix=rna_csv_file_path,
                cell_labels=labels_csv_file_path
            )
            ipd.fit_all_antibodies(model=model)
            ipd.normalize_all_antibodies(sig_expr_threshold=sig_expr_threshold, 
                                        bg_expr_threshold=bg_expr_threshold, 
                                        p_threshold=p_threshold, 
                                        bg_cell_z_score=bg_cell_z_score)
            ipd.normalized_counts.to_csv(output_csv)
            print(f"Normalized counts saved to {output_csv}")

            print("Generating UMAPs...")
            double_UMAP = combine_umap_plots(IPD=ipd).savefig(output_umap)
            print("UMAP saved to:", output_umap)
    
    elif scanpy_path and not protein_csv_file_path:
        # Scanpy with labels
        if scanpy_labels:    
            pbmc = sc.read_h5ad(scanpy_path)
            pbmc_ipd = ImmunoPhenoData(scanpy=pbmc, scanpy_labels=scanpy_labels)
            pbmc_ipd.fit_all_antibodies(model=model)
            pbmc_ipd.normalize_all_antibodies(sig_expr_threshold=sig_expr_threshold, 
                                            bg_expr_threshold=bg_expr_threshold, 
                                            p_threshold=p_threshold, 
                                            bg_cell_z_score=bg_cell_z_score)
            pbmc_ipd.normalized_counts.to_csv(output_csv)
            print(f"Normalized counts saved to {output_csv}")

            print("Generating UMAPs...")
            double_UMAP = combine_umap_plots(IPD=pbmc_ipd).savefig(output_umap)
            print("UMAP saved to:", output_umap)
        
        # Scanpy without labels
        else:
            pbmc = sc.read_h5ad(scanpy_path)
            pbmc_ipd = ImmunoPhenoData(scanpy=pbmc)
            pbmc_ipd.fit_all_antibodies(model=model)
            pbmc_ipd.normalize_all_antibodies(sig_expr_threshold=sig_expr_threshold, 
                                            bg_expr_threshold=bg_expr_threshold, 
                                            p_threshold=p_threshold, 
                                            bg_cell_z_score=bg_cell_z_score)
            pbmc_ipd.normalized_counts.to_csv(output_csv)
            print(f"Normalized counts saved to {output_csv}")

            print("Generating UMAPs...")
            double_UMAP = combine_umap_plots(IPD=pbmc_ipd).savefig(output_umap)
            print("UMAP saved to:", output_umap)
    else:
        print("Error: Invalid combination of arguments. Please provide the necessary data files and parameters.")

def process_args(args):
    params = {
        "model": args.model,
        "sig_expr_threshold": args.sig_expr_threshold,
        "bg_expr_threshold": args.bg_expr_threshold,
        "p_threshold": args.p_threshold,
        "bg_cell_z_score": args.bg_cell_z_score
    }

    # Assign conditional arguments based on provided inputs
    if args.protein:
        params["protein_csv_file_path"] = args.protein
    if args.rna:
        params["rna_csv_file_path"] = args.rna
    if args.labels:
        params["labels_csv_file_path"] = args.labels
    if args.scanpy_path:
        params["scanpy_path"] = args.scanpy_path
    if args.scanpy_labels:
        params["scanpy_labels"] = args.scanpy_labels
    if args.output_csv:
        params["output_csv"] = args.output_csv
    if args.umap:
        params["output_umap"] = args.umap

    # Call main function with the prepared parameters
    main(**params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ImmunoPhenoData and save normalized counts to CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
        usage='''
        python normalization.py [required] <model> [optional] <protein> <rna> <labels> <scanpy_path> <scanpy_labels>
        
        Required Parameters:
        -m, --model                 Type of model for fit_all_antibodies function. "nb" or "gaussian"
        
        Optional Parameters:
        -f1, --protein              Path to the .csv file containing protein counts.
        -f2, --rna                  Path to the .csv file containing gene/rna counts.
        -l, --labels                Path to the .csv file containing the cell labels.
        -sp, --scanpy_path          Path to the .h5ad file
        -sl, --scanpy_labels        Labels field name inside scanpy object
        -u, --umap                  Custom name of output PNG containing regular and normalized UMAP
        -o, --output_csv            Custom name of output CSV containing normalized counts
        -s, --sig_expr_threshold    Signal expression threshold (default: 1)
        -b, --bg_expr_threshold     Background expression threshold (default: 0)
        -p, --p_threshold           P-value threshold (default: 0.05)
        -z, --bg_cell_z_score       Background cell z-score (default: 10.0)
        '''
    )

    # Required argument
    parser.add_argument("-m", "--model", type=str, required=True, help="Model for fit_all_antibodies function.")
    parser.add_argument("-u", "--umap", type=str, required=True, help="Custom name of output PNG for UMAPs.")
    parser.add_argument("-o", "--output_csv", type=str, required=True, help="Custom name of output CSV for normalized counts.")
    
    # Optional Arguments
    parser.add_argument("-f1", "--protein", type=str, help="Path to the protein .csv file.")
    parser.add_argument("-f2", "--rna", type=str, help="Path to the rna .csv file.")
    parser.add_argument("-l", "--labels", type=str, help="Path to the labels .csv file.")
    parser.add_argument("-sp", "--scanpy_path", type=str, help="Path to the .h5ad file.")
    parser.add_argument("-sl", "--scanpy_labels", type=str, help="Labels for scanpy data.")
    
    # Parameters for normalization filtering
    parser.add_argument("-s", "--sig_expr_threshold", type=float, default=1, help="Signal expression threshold")
    parser.add_argument("-b", "--bg_expr_threshold", type=float, default=0, help="Background expression threshold")
    parser.add_argument("-p", "--p_threshold", type=float, default=0.05, help="P-value threshold")
    parser.add_argument("-z", "--bg_cell_z_score", type=float, default=10, help="Background cell z-score")

    args = parser.parse_args()
    process_args(args)