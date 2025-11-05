import argparse
import pandas as pd
import scanpy as sc
import copy
import umap
from immunopheno.connect import ImmunoPhenoDB_Connect
from immunopheno.data_processing import ImmunoPhenoData
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import io

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
            fig, ax = plt.subplots(figsize=(18, 8))
            sns.scatterplot(
                x=norm_projections[:, 0],
                y=norm_projections[:, 1],
                s=7,
                ax=ax,
                legend=False,
            )
            ax.set_title("Normalized Protein Expression UMAP, No Annotations")
            fig.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.1)
            return fig

        # Un-normalized UMAP without cell labels
        elif IPD._cell_labels is None and not normalized:
            # Create Seaborn plot
            fig, ax = plt.subplots(figsize=(18, 8))
            sns.scatterplot(
                x=raw_projections[:, 0],
                y=raw_projections[:, 1],
                s=7,
                ax=ax,
                legend=False
            )
            ax.set_title("Regular Protein Expression UMAP, No Annotations")
            fig.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.1)
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
            fig, ax = plt.subplots(figsize=(18, 8))
            sns.scatterplot(
                x=norm_projections[:, 0],
                y=norm_projections[:, 1],
                hue=norm_types,
                palette="tab20",
                s=7,
                ax=ax,
                legend="full"
            )
            ax.legend(title="Cell Type", 
                      bbox_to_anchor=(1.02, 1), 
                      loc='upper left', 
                      markerscale=2,
                      fontsize=9)
            ax.set_title("Normalized Protein Expression UMAP, ImmunoPheno Annotations")
            fig.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.1)
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
            fig, ax = plt.subplots(figsize=(18, 8))
            sns.scatterplot(
                x=raw_projections[:, 0],
                y=raw_projections[:, 1],
                hue=raw_types,
                palette="tab20",
                s=7,
                ax=ax,
                legend="full"
            )
            ax.legend(title="Cell Type", 
                      bbox_to_anchor=(1.02, 1), 
                      loc='upper left', 
                      markerscale=2,
                      fontsize=9)
            ax.set_title("Regular Protein Expression UMAP, ImmunoPheno Annotations")
            fig.subplots_adjust(left=0.05, right=0.75, top=0.9, bottom=0.1)
            return fig

def plot_spatial(spatial_df,
                 normalized=False,
                 x_label="x_coord",
                 y_label="y_coord",):
    # 1. Increased the figure width to make room for the legend.
    fig, ax = plt.subplots(figsize=(18, 12))

    if normalized:
        sns.scatterplot(
                x=spatial_df[x_label],
                y=spatial_df[y_label],
                palette="tab20",
                s=1,
                hue="celltype", # using transferred labels
                ax=ax,
                legend="full",
                data=spatial_df)
        ax.legend(title="Cell Type",
                  bbox_to_anchor=(1.01, 1), 
                  loc='upper left', 
                  borderaxespad=0.,
                  markerscale=3,
                  fontsize=9)
        ax.set_title("ImmunoPheno Annotations")
        
    else:
        sns.scatterplot(
                    x=spatial_df[x_label],
                    y=spatial_df[y_label],
                    palette="tab20",
                    s=1,
                    ax=ax,
                    legend=False) # No legend for this plot
        ax.set_title("No Annotations")

    # 2. Removed plt.tight_layout() and manually adjust subplot parameters.
    # This ensures a consistent plot area size, leaving space for the legend on the right.
    fig.subplots_adjust(left=0.08, right=0.75, top=0.92, bottom=0.08)
    
    return fig

def combine_umap_images(fig1, fig2, output_name):
    """Combines two in-memory UMAP images vertically into a single PNG.

    Args:
        fig1 (plt figure): First UMAP figure (e.g., regular UMAP).
        fig2: (plt figure): Second UMAP figure (e.g., normalized UMAP).
        output_name (str): Output file name for the combined image.
    """
    # Save each Matplotlib figure as an in-memory PNG image
    img_bytes1 = io.BytesIO()
    fig1.savefig(img_bytes1, format="png")
    img_bytes1.seek(0)  # Reset pointer to the beginning of the BytesIO object
    img1 = Image.open(img_bytes1)

    img_bytes2 = io.BytesIO()
    fig2.savefig(img_bytes2, format="png")
    img_bytes2.seek(0)  # Reset pointer to the beginning of the BytesIO object
    img2 = Image.open(img_bytes2)

    # Determine the combined image height and maximum width
    combined_height = img1.height + img2.height
    max_width = max(img1.width, img2.width)

    # Create a new blank image with the combined dimensions
    combined_img = Image.new("RGB", (max_width, combined_height))

    # Paste each image into the combined image, one on top of the other
    combined_img.paste(img1, (0, 0))
    combined_img.paste(img2, (0, img1.height))

    # Save the combined image
    combined_img.save(output_name, "PNG")

def main(protein_csv_file_path=None,
         spatial_coord_csv_file_path=None,
         antibody_spreadsheet_csv_file_path=None,
         model="gaussian",
         sig_expr_threshold=1,
         bg_expr_threshold=0,
         antibody_matching_option=2,
         imputation_factor=0.5,
         x_label="x_coord",
         y_label="y_coord",
         k_find_nn=40,
         k_find_anchor=20,
         k_transfer_matrix=40,
         num_chunks=1):
    """
    Given only protein, spatial, and antibody spreadsheet.
    Normalize the protein data, and perform annotation transfer
    with options to choose for:
        Antibody Matching Option, where "1" is "Clone" and "2" is Antibody Target
        Rho Imputation
        Number of chunks for annotations (will be single core, but multiple runs)

    Output:
        CSV of all cell labels
        Before-and-after plot of annotations
    """
    # Assuming we are working with spatial mIHC data, 
    # Output:
    #   1. The list of labels
    #   2. Before/after spatial plots
    if protein_csv_file_path and spatial_coord_csv_file_path:
        # First create an IPD object to normalize
        print("Fitting mixture models...")
        ipd = ImmunoPhenoData(protein_matrix=protein_csv_file_path,
                            spreadsheet=antibody_spreadsheet_csv_file_path)
        ipd.fit_all_antibodies(model=model)
        print("Normalizing data...")
        ipd.normalize_all_antibodies(sig_expr_threshold=sig_expr_threshold,
                                    bg_expr_threshold=bg_expr_threshold)
        

        # Now with the normalized data, begin the annotation transfer
        print("Beginning annotation transfer...")
        cxn = ImmunoPhenoDB_Connect("http://www.immunopheno.org")
        ipd_new = cxn.run_stvea(IPD=ipd,
                                parse_option=antibody_matching_option,
                                rho=imputation_factor,
                                k_find_nn=k_find_nn,
                                k_find_anchor=k_find_anchor,
                                k_transfer_matrix=k_transfer_matrix,
                                num_chunks=num_chunks)
        
        # Retrieve the full list of labels (unnormalized) and output into a csv
        ipd_new._cell_labels.to_csv("immunopheno_labels.csv")
        print("Predicted labels saved to: immunopheno_labels.csv")
        
        # Now get the transferred annotations
        spat = pd.read_csv(spatial_coord_csv_file_path, index_col=0)
        spat_copy = copy.deepcopy(spat)
        spat_copy.loc[ipd_new.labels.index, "celltype"] = ipd_new.labels["celltype"]
        # spat_copy.to_csv("spat_copy_celltype.csv")
        fig_transferred = plot_spatial(spatial_df=spat_copy,
                                       normalized=True,
                                       x_label=x_label,
                                       y_label=y_label)
        
        # Plot the regular spatial plot without any annotations
        fig_original = plot_spatial(spatial_df=spat,
                                    normalized=False,
                                    x_label=x_label,
                                    y_label=y_label)
        
        # Combine these two into a single image now
        combine_umap_images(fig_original, fig_transferred, "combined_plot.png")
        print("Spatial plots saved to: combined_plot.png")


    # If working with non spatial data (just plain proteomic data), 
    # Output:
    #   1. The list of labels
    #   2. Before/after UMAPs
    elif protein_csv_file_path and not spatial_coord_csv_file_path:
        # First create an IPD object to normalize
        print("Fitting mixture models...")
        ipd = ImmunoPhenoData(protein_matrix=protein_csv_file_path,
                            spreadsheet=antibody_spreadsheet_csv_file_path)
        ipd.fit_all_antibodies(model=model)
        print("Normalizing data...")
        ipd.normalize_all_antibodies(sig_expr_threshold=sig_expr_threshold,
                                    bg_expr_threshold=bg_expr_threshold)
        
        # Now with the normalized data, begin the annotation transfer
        print("Beginning annotation transfer...")
        cxn = ImmunoPhenoDB_Connect("http://www.immunopheno.org")
        ipd_new = cxn.run_stvea(IPD=ipd,
                                parse_option=antibody_matching_option,
                                rho=imputation_factor,
                                k_find_nn=k_find_nn,
                                k_find_anchor=k_find_anchor,
                                k_transfer_matrix=k_transfer_matrix,
                                num_chunks=num_chunks)
        
        # Retrieve the list of labels and output into a csv
        ipd_new._cell_labels.to_csv("immunopheno_labels.csv")
        print("Predicted labels saved to: immunopheno_labels.csv")

        # Plot both regular and normalized UMAPs and output as PNG
        print("Generating UMAPs...")
        regular_umap_fig = plot_UMAP(IPD=ipd, normalized=False)
        normalized_umap_fig = plot_UMAP(IPD=ipd_new, normalized=False)
        # Combine and save as a single vertically stacked image
        combine_umap_images(regular_umap_fig, normalized_umap_fig, "combined_plot.png")
        print("UMAP saved to: combined_plot.png")
    
def process_args(args):
    params = {
        "model": args.model,
        "sig_expr_threshold": args.sig_expr_threshold,
        "bg_expr_threshold": args.bg_expr_threshold,
    }

    # Assign conditional arguments based on provided inputs
    if args.protein:
        params["protein_csv_file_path"] = args.protein
    if args.spatial:
        params["spatial_coord_csv_file_path"] = args.spatial
    if args.antibodies_csv:
        params["antibody_spreadsheet_csv_file_path"] = args.antibodies_csv
    if args.matching:
        params["antibody_matching_option"] = args.matching
    if args.imputation:
        params["imputation_factor"] = args.imputation
    if args.chunks:
        params["num_chunks"] = args.chunks
    if args.x_label:
        params["x_label"] = args.x_label
    if args.y_label:
        params["y_label"] = args.y_label
    # Call main function with the prepared parameters
    main(**params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process ImmunoPhenoData and save normalized counts to CSV.",
        formatter_class=argparse.RawTextHelpFormatter,
        usage='''
        python annotation.py [required] <model> <protein> <antibodies_csv> [optional] <chunks> <matching> <spatial>
        
        Required Parameters:
        -p, --protein               Path to the .csv file containing protein counts.
        -a, --antibodies_csv        CSV containing the RRIDs for each antibody in the proteomic query dataset
        -m, --model                 Type of model for fit_all_antibodies function. "nb" or "gaussian"
    
        Optional Parameters:
        -c, --chunks                Number of chunks to evenly split the data when annotating large datasets (50k+ cells)
        -i, --imputation            Imputation tuning parameter for insufficient data. Default: 0.5. Reduce if insufficient antibodies present.
        -o, --matching              Matching algorithm for finding antibodies between query and reference. Defaults to "2", or antibody target.
        -sp, --spatial              CSV containing spatial coordinates for each cell
        -x, --x_label               Name of X coordinate column, if spatial data is provided
        -y, --y_label               Name of Y coordinate column, if spatial data is provided
        -s, --sig_expr_threshold    Signal expression threshold (default: 1)
        -b, --bg_expr_threshold     Background expression threshold (default: 0)
        '''
    )

    # Required arguments
    parser.add_argument("-m", "--model", type=str, required=True, help="Model for fit_all_antibodies function.")
    parser.add_argument("-p", "--protein", type=str, required=True, help="Path to the protein .csv file.")
    parser.add_argument("-a", "--antibodies_csv", type=str, required=True, help="Path to antibody spreadsheet .csv file")

    # Parameters for annotation transfer step
    parser.add_argument("-sp", "--spatial", type=str, required=False, help="Path to spatial coordinates .csv file")
    parser.add_argument("-c", "--chunks", type=int, default=1, help="Number of chunks to evenly split the data")
    parser.add_argument("-i", "--imputation", type=float, default=0.5, help="Imputation tuning parameter for handling missing data")
    parser.add_argument("-o", "--matching", type=int, default=2, help="Find reference antibodies by either Clone (1) or Antibody Target (2)")
    parser.add_argument("-x", "--x_label", type=str, default="x_coord", help="Name of column containing x coordinates in spatial data")
    parser.add_argument("-y", "--y_label", type=str, default="y_coord", help="Name of column containing y coordinates in spatial data")

    # Parameters for normalization filtering
    parser.add_argument("-s", "--sig_expr_threshold", type=float, default=1, help="Signal expression threshold")
    parser.add_argument("-b", "--bg_expr_threshold", type=float, default=0, help="Background expression threshold")

    args = parser.parse_args()
    process_args(args)