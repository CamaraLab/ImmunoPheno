import argparse
import pandas as pd
import scanpy as sc
from immunopheno.data_processing import ImmunoPhenoData

import umap
import seaborn as sns
import matplotlib.pyplot as plt

def combine_umap_plots(IPD, **kwargs):
    """
    Generates and combines unnormalized and normalized UMAP plots into a single figure,
    correctly handling differences in cell counts and using constrained_layout to prevent overlap.

    Args:
        IPD: ImmunoPhenoData object containing protein and label data.
        random_state (int): The random state for UMAP for reproducibility.
        **kwargs: Optional arguments to pass to the UMAP class constructor.

    Returns:
        matplotlib.figure.Figure: The combined figure object.
    """
    # Create a figure with two subplots using the robust constrained_layout engine
    fig_final, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 16), constrained_layout=True)

    # If no UMAP or kwargs are different, generate a new one and store in class
    umap_plot = umap.UMAP(random_state=42, **kwargs)

    norm_projections = umap_plot.fit_transform(IPD.normalized_counts)
    raw_projections = umap_plot.fit_transform(IPD.protein)
    
    # ---- Make unnormalized UMAP first
    if IPD._cell_labels is None:
        sns.scatterplot(
            x=raw_projections[:, 0],
            y=raw_projections[:, 1],
            s=7,
            ax=ax1,
        )
        ax1.set_title("Regular Protein Expression UMAP")
        plt.tight_layout()

    if IPD._cell_labels is not None:
        # Check the number of columns
        num_columns = IPD._cell_labels.shape[1]
        # Check if there is at least one column and if the second column is not empty
        if num_columns > 1 and not IPD._cell_labels.iloc[:, 1].isnull().all():
            # Use the values from the second column
            raw_types = IPD._cell_labels.iloc[:, 1].tolist()
        else:
            # If there is no second column or it is empty, use the values from the first column
            raw_types = IPD._cell_labels.iloc[:, 0].tolist()

        sns.scatterplot(
            x=raw_projections[:, 0],
            y=raw_projections[:, 1],
            hue=raw_types,
            palette="tab20",
            s=7,
            ax=ax1,
            legend="full"
        )
        ax1.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        ax1.set_title("Regular Protein Expression UMAP")
        plt.tight_layout()

    # ---- Normalized UMAP without cell labels
    if IPD.labels is None:
        sns.scatterplot(
            x=norm_projections[:, 0],
            y=norm_projections[:, 1],
            s=7,
            ax=ax2,
        )
        ax2.set_title("Normalized Protein Expression UMAP")
        plt.tight_layout()

    # Normalized UMAP plot with cell labels
    if IPD.labels is not None:
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
        # fig, ax = plt.subplots(figsize=(16, 6))
        sns.scatterplot(
            x=norm_projections[:, 0],
            y=norm_projections[:, 1],
            hue=norm_types,
            palette="tab20",
            s=7,
            ax=ax2,
            legend="full"
        )
        ax2.legend(title="Cell Type", bbox_to_anchor=(1.05, 1), loc='upper left', markerscale=2)
        ax2.set_title("Normalized Protein Expression UMAP")
        plt.tight_layout()

    return fig_final

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
            # Generate the figure
            combined_fig = combine_umap_plots(IPD=ipd)
            # Save the figure, adjusting the size to fit the legends
            combined_fig.savefig(output_umap, bbox_inches='tight')
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
            # Generate the figure
            combined_fig = combine_umap_plots(IPD=ipd)
            # Save the figure, adjusting the size to fit the legends
            combined_fig.savefig(output_umap, bbox_inches='tight')
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
            # Generate the figure
            combined_fig = combine_umap_plots(IPD=ipd)
            # Save the figure, adjusting the size to fit the legends
            combined_fig.savefig(output_umap, bbox_inches='tight')
            print("UMAP saved to:", output_umap)
    
    elif scanpy_path and not protein_csv_file_path:
        # Scanpy with labels
        if scanpy_labels:    
            sc_ipd = sc.read_h5ad(scanpy_path)
            ipd = ImmunoPhenoData(scanpy=sc_ipd, scanpy_labels=scanpy_labels)
            ipd.fit_all_antibodies(model=model)
            ipd.normalize_all_antibodies(sig_expr_threshold=sig_expr_threshold, 
                                            bg_expr_threshold=bg_expr_threshold, 
                                            p_threshold=p_threshold, 
                                            bg_cell_z_score=bg_cell_z_score)
            ipd.normalized_counts.to_csv(output_csv)
            print(f"Normalized counts saved to {output_csv}")

            print("Generating UMAPs...")
            # Generate the figure
            combined_fig = combine_umap_plots(IPD=ipd)
            # Save the figure, adjusting the size to fit the legends
            combined_fig.savefig(output_umap, bbox_inches='tight')
            print("UMAP saved to:", output_umap)
        
        # Scanpy without labels
        else:
            sc_ipd = sc.read_h5ad(scanpy_path)
            ipd = ImmunoPhenoData(scanpy=sc_ipd)
            ipd.fit_all_antibodies(model=model)
            ipd.normalize_all_antibodies(sig_expr_threshold=sig_expr_threshold, 
                                            bg_expr_threshold=bg_expr_threshold, 
                                            p_threshold=p_threshold, 
                                            bg_cell_z_score=bg_cell_z_score)
            ipd.normalized_counts.to_csv(output_csv)
            print(f"Normalized counts saved to {output_csv}")

            print("Generating UMAPs...")
            # Generate the figure
            combined_fig = combine_umap_plots(IPD=ipd)
            # Save the figure, adjusting the size to fit the legends
            combined_fig.savefig(output_umap, bbox_inches='tight')
            print("UMAP saved to:", output_umap)
    else:
        print("Error: Invalid combination of arguments. Please provide the necessary data files and parameters.")

def process_args(args):
    params = {
        "model": args.model,
        "sig_expr_threshold": args.sig_expr_threshold,
        "bg_expr_threshold": args.bg_expr_threshold,
        # "p_threshold": args.p_threshold,
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
        # -p, --p_threshold           P-value threshold (default: 0.05)
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
    # parser.add_argument("-p", "--p_threshold", type=float, default=0.05, help="P-value threshold")
    parser.add_argument("-z", "--bg_cell_z_score", type=float, default=10, help="Background cell z-score")

    args = parser.parse_args()
    process_args(args)