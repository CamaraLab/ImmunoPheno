import argparse
import pandas as pd
import json
from immunopheno.connect import ImmunoPhenoDB_Connect

def main(target=None,
         background=None,
         tissue=None,
         panel_size=5,
         imputation_factor=0.35,
         plot_gates=True,
         plot_decision_tree=True):
    """
    This should generate 4 things
    1. CSV of the optimal antibodies
    2. PNG of the gates produced using those antibodies
    3. PNG of the decision tree produced corresponding to the gates
    4. JSON of the paths, yield, and purity from the gates
    """
    print("Connecting to ImmunoPheno database...")
    cxn = ImmunoPhenoDB_Connect("http://www.immunopheno.org")
    
    print(f"Finding optimal panel for target(s): {target}")
    if background:
        print(f"Distinguishing from background: {background}")
    if tissue:
        print(f"Restricted to tissues: {tissue}")

    optimal_ab, path_yield_purity, gates = cxn.optimal_antibody_panel(
        target=target,
        background=background,
        tissue=tissue, 
        panel_size=panel_size,
        rho=imputation_factor,
        plot_gates=plot_gates, 
        plot_decision_tree=plot_decision_tree,
        merge_option=1)
    
    # --- Save all results to predictable filenames ---
    print("Saving results...")
    # The decision tree is automatically saved by the function to 'decision_tree.png'
    optimal_ab.to_csv("optimal_ab_table.csv")
    gates.savefig("gates.png")
    with open("path_yield_purity.json", "w") as file:
        json.dump(path_yield_purity, file, indent=4)
    print("Script finished successfully.")


def process_args(args):
    params = {
        "target": args.target,
        "panel_size": args.panel_size,
        "imputation_factor": args.imputation,
    }

    if args.background:
        params["background"] = args.background
    if args.tissue:
        params["tissue"] = args.tissue
    
    # These are always true for the Galaxy tool
    params["plot_gates"] = True
    params["plot_decision_tree"] = True

    main(**params)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate optimal antibody panel, decision tree, predicted gates, and purity/yield.",
    )

    # The 'nargs' argument tells argparse to accept one or more values for this flag
    # and store them in a list.
    parser.add_argument(
        "-t", "--target", 
        type=str, 
        nargs='+',
        required=True, 
        help="One or more target Cell Ontology IDs (e.g., CL:0000940 CL:0000818)"
    )
    parser.add_argument(
        "-b", "--background", 
        type=str, 
        nargs='+',  # Also accept a list for the background
        required=False, 
        help="One or more background Cell Ontology IDs to distinguish against"
    )

    parser.add_argument(
        "-o", "--tissue",
        type=str,
        nargs="+",
        required=False,
        help="One or more tissue Brenda Ontology IDs (e.g., BTO:0000141, BTO:0004122)"
    )

    parser.add_argument(
        "-p", "--panel_size", 
        type=int, 
        required=True, 
        help="The desired number of antibodies in the panel."
    )
    parser.add_argument(
        "-i", "--imputation", 
        type=float, 
        default=0.35, 
        help="Imputation tuning parameter for handling missing data."
    )

    args = parser.parse_args()
    process_args(args)