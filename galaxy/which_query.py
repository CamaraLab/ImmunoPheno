# --- START OF FILE which_query.py (Corrected) ---

import argparse
import pandas as pd
from immunopheno.connect import ImmunoPhenoDB_Connect

def which_antibodies(query):
    print(f"Querying for antibodies with term: '{query}'")
    cxn = ImmunoPhenoDB_Connect("http://www.immunopheno.org")
    result = cxn.which_antibodies(query)
    return result

def which_celltypes(query):
    print(f"Querying for cell types with term: '{query}'")
    cxn = ImmunoPhenoDB_Connect("http://www.immunopheno.org")
    result = cxn.which_celltypes(query)
    return result

def which_experiments(query):
    print(f"Querying for experiments with term: '{query}'")
    cxn = ImmunoPhenoDB_Connect("http://www.immunopheno.org")
    result = cxn.which_experiments(query)
    return result

def main(query_type, query_term):
    """
    Main logic function. Takes the type of query and the term,
    calls the appropriate function, and saves the result.
    """
    if query_type == "antibodies":
        result_df = which_antibodies(query_term)
    elif query_type == "celltypes":
        result_df = which_celltypes(query_term)
    elif query_type == "experiments":
        result_df = which_experiments(query_term)
    else:
        # This case should not be reached if called from argparse
        raise ValueError("Invalid query type specified.")

    # Save the result to a predictable filename
    result_df.to_csv("query_result.csv")
    print("Query successful. Results saved to query_result.csv")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Queries the ImmunoPheno database for antibodies, cell types, or experiments."
    )

    # A single, required argument for the query term
    parser.add_argument(
        "-q", "--query_term", 
        required=True, 
        type=str, 
        help="The word/phrase to search for (e.g., 'CD4', 'mucosal', 'bone marrow')."
    )

    # A mutually exclusive group to ensure the user can only pick one query type
    query_type_group = parser.add_mutually_exclusive_group(required=True)
    query_type_group.add_argument(
        "-a", "--antibodies", 
        action='store_true', 
        help="Query for antibodies."
    )
    query_type_group.add_argument(
        "-c", "--celltypes", 
        action='store_true', 
        help="Query for cell types."
    )
    query_type_group.add_argument(
        "-e", "--experiments", 
        action='store_true', 
        help="Query for experiments."
    )

    args = parser.parse_args()

    # Determine which query type was selected and call main
    if args.antibodies:
        main(query_type="antibodies", query_term=args.query_term)
    elif args.celltypes:
        main(query_type="celltypes", query_term=args.query_term)
    elif args.experiments:
        main(query_type="experiments", query_term=args.query_term)