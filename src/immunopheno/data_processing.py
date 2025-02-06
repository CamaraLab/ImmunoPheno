import warnings
import logging
import csv
import multiprocessing
import multiprocess
import anndata
import requests
import math
import numpy as np
import scipy
import scipy.stats as ss
import statsmodels.api as sm
import pandas as pd
from importlib.resources import files
from tqdm.autonotebook import tqdm
from sklearn.linear_model import LinearRegression
from .models import _gmm_results, _nb_mle_results

def _load_adt(protein: str | pd.DataFrame) -> pd.DataFrame:
    """
    Loads the protein data from CSV or pandas DataFrame

    Parameters:
      protein (str or pd.DataFrame): csv file path or dataframe where:
        rows = cells, cols = protein

    Returns:
      protein_df_copy (Pandas DataFrame):
    """

    if isinstance(protein, str):
        protein_df = pd.read_csv(protein, sep=",", index_col=[0]) # no transpose
    else:
        protein_df = protein

    return protein_df

def _generate_range(size: int,
                   interval_size: int = 2000,
                   start: int = 0) -> list:
    """
    Splits a range of numbers into partitions based on a
    provided interval size. Partitions will only contain
    the begin and end of a range. Ex: [[0, 10], [10, 20]]

    Parameters:
        size (int): total size of the range
        interval_size (int): desired partition sizes
        start (int): index to begin partitioning the range

    Returns:
        start_stop_ranges (list): list of lists, containing
            the start and end of a partition.
            ex: [[0, 20], [20, 40], [40, 60]]
    """

    if interval_size <= 1:
        raise ValueError("Chunk or partition size must be greater than 1.")

    iterable_range = [i for i in range(size + 1)]

    end = max(iterable_range)
    step = interval_size

    start_stop_ranges = []

    for i in range(start, end, step):
        x = i
        start_stop_ranges.append([iterable_range[x:x+step][0],
                                  iterable_range[x:x+step+1][-1]])

    return start_stop_ranges

def _read_csv(file_path: str, chunk_range=None):
    """
    Reads in a (large) csv file using a parser. This
    avoid having to load it into memory all at once.

    Parameters:
        file_path (str): file path to csv file
        chunk_range (list): specific rows to parse

    Returns:
        row (generator object): generator containing all/specific
        rows in csv as a list
    """

    if chunk_range is not None:
    # Use set for O(1) average lookup
        col_indices = set(range(chunk_range[0], chunk_range[1]))

    with open(file_path, "r") as f:
        reader = csv.reader(f)
        
        if chunk_range is not None:
            for i, column in enumerate(reader):
                if i in col_indices:
                    yield list(column)
        else:
            for column in reader:
                yield list(column)

def _umi_generator(file_path: str, chunk_range=None):
    """
    Reads in an RNA csv file and produces a generator containing
    UMI counts for each cell as a list

    Parameters:
        file_path (str): file path to csv file
        chunk_range (list): specific rows to parse

    Returns:
        row (generator object): generator containing UMI counts for
            each gene, for a single cell. Stored as a list
    """

    for row in _read_csv(file_path, chunk_range):
        if row[0].lower() != "":
            yield [int(x) for x in row[1:]]

def _cell_name_generator(file_path: str):
    """
    Reads in an RNA csv file and produces a generator containing
    cell name (barcodes)

    Parameters:
        file_path (str): file path to csv file

    Returns:
        row (generator object): generator containing all cell barcodes
            Stored as a list
    """

    for row in _read_csv(file_path):
        if row[0].lower() != "":
            yield row[0]

def _clean_chunks(chunk_list: list) -> list:
    """
    Merges any chunks in a list that only contain one element with
    the previous chunk. This is because SingleR requires at least
    two cells at once to run. This may also reduce overhead by
    reducing the number of processes to allocate.

    Parameters:
        chunk_list (list): list of lists containing start and end values

    Returns:
        chunk_list (list): updated list of lists without any lists
            that contain a single element
    """
    final_chunk = chunk_list[-1]
    size_final_chunk = final_chunk[1] - final_chunk[0]

    if size_final_chunk == 1:
        # Update the second to last chunk with the last chunk
        # "Merge" by adding the last chunk to the second to last chunk
        # and delete the last chunk
        chunk_list[-2][1] = chunk_list[-2][1] + 1
        chunk_list.pop()

    return chunk_list

def _load_rna_parallel(gene_filepath: str) -> pd.DataFrame:
    """
    Loads in a large RNA csv file using parallelized batch processing.

    Large RNA dataframes will be divided into large 'chunks', which
    are divided into smaller 'partitions'.

    Chunks will be evaluated sequentially, while partitions inside a chunk
    will be evaluated in parallel as smaller pandas DataFrames.

    Partitions will be concatenated together at the end to create a
    sparse pandas DataFrame.

    Parameters:
        gene_filepath (str): csv file path with rows (genes) x columns (cells)

    Returns:
        results_df (pd.DataFrame): sparse DataFrame with rows (cells) x columns (genes)
    """

    # Create a generator to get all the gene names (stored as the columns)
    gene_name_generator = _read_csv(gene_filepath)
    gene_names = [next(gene_name_generator, None) for _ in range(1)][0][1:]
    
    # Create a generator to get all cell names
    cell_name_generator = _cell_name_generator(gene_filepath)
    cell_names = list(cell_name_generator)

    # Default chunk size will be num cells // 2
    chunk_size = len(cell_names) // 2

    # Default partition size will be chunk_size // 3
    partition_size = (chunk_size // 3) 

    # Generate all the ranges for cells in the dataset
    cell_ranges = _clean_chunks(_generate_range(len(cell_names) + 1, # add one to include last row
                                               start=0,
                                               interval_size=chunk_size))

    dataframe_chunks = []
    def process_section(chunk):
        umi_chunk = _umi_generator(gene_filepath, chunk)
        umi_chunk_df = pd.DataFrame.sparse.from_spmatrix(scipy.sparse.csr_matrix(list(umi_chunk)))
        # return umi_chunk_df.T # less expensive to transpose here compare to the end
        return umi_chunk_df # no transpose, assume user provides cells as rows, ab as columns

    for chunk in cell_ranges:
        partition_ranges = _clean_chunks(_generate_range(chunk[1],
                                                       interval_size=partition_size,
                                                       start=chunk[0]))
        # Create a multiprocessing pool
        with multiprocess.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(process_section, partition_ranges)

        for rna_df in results:
            dataframe_chunks.append(rna_df)

    # Combine all DataFrames
    results_df = pd.concat(dataframe_chunks, ignore_index=True, axis=0) # axis = 0 concatenates vertically

    # Add in Index and Columns
    cell_index = pd.Index(cell_names)
    results_df.set_index(cell_index, inplace=True)
    results_df.columns = gene_names

    return results_df

def _load_rna(gene: str | pd.DataFrame) -> pd.DataFrame:
    """
    Loads in RNA data from a CSV or pandas DataFrame

    Parameters:
        gene (str or pd.DataFrame): csv file path with rows (genes) x columns (cells)

    Returns:
        rna_df (pd.DataFrame): RNA dataframe with rows (cells) x columns (genes)
    """

    # For csv file paths
    if isinstance(gene, str):
        # If the number of cells is <= 20k, we can use pandas directly to load
        # Create a generator to get all the column names
        cell_name_generator = _cell_name_generator(gene)
        cell_names = list(cell_name_generator)

        if len(cell_names) <= 20000:
            rna_df = pd.read_csv(gene, sep=",", index_col=[0]) # assume user provides cells as rows

        # Otherwise, load in parallel
        else:
            rna_df = _load_rna_parallel(gene)
    else:
        rna_df = gene

    return rna_df

def _singleR_rna(rna: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out genes (columns) that are used when
    running SingleR

    Parameters:
        rna (pd.DataFrame): RNA dataframe

    Returns:
        Pandas dataframe with columns only found in the
        Human Primary Cell Atlas (SingleR)
    """

    # Retrieve list of genes used in SingleR
    # We will filter out unused genes from the RNA dataset that are not in that list
    hpca_genes_path = str(files('immunopheno.data').joinpath('hpca_genes.txt'))
    hpca_genes = set(line.strip() for line in open(hpca_genes_path))

    rna_column_genes = pd.Index([x.upper() for x in rna.columns])
    return rna.loc[:, rna_column_genes.isin([x.upper() for x in hpca_genes])]

def _read_antibodies(csv_file: str) -> list:
    """
    Parses a spreadsheet containing antibody IDs

    Parameters:
        csv_file (str): name of csv file containing antibody names

    Returns:
        antibodies (list): list of lists with ab_name and antibody ids
        ex: [["CD4", "AB_XXXXXXX"], ["CD5", "AB_XXXXXXX"]]
    """
    antibodies = []

    with open(csv_file, 'r') as csv_file:
        lines = csv_file.readlines()
        num_lines = len(lines)
        ab_index = next((i for i, line in enumerate(lines) if 'Antibody table' in line), None) + 1

        if ab_index is not None:
            while ab_index < num_lines:
                ab = lines[ab_index].strip().split(",", maxsplit=1)
                ab_name_strip = ab[0].strip()
                ab_id_strip = ab[1].strip()
                antibodies.append([ab_name_strip, ab_id_strip])
                ab_index += 1
        else:
            print("'Antibody table' not found in any line.")
    
    return antibodies

def _target_ab_dict(antibody_pairs: list) -> dict:
    """
    Creates a dictionary of antibody names with their IDs

    Parameters:
        antibody_pairs (list): list of antibody pairs from _read_antibodies().
            Ex: [["CD4", "AB_XXXXXXX"], ["CD5", "AB_XXXXXXX"]]

    Returns:
        target_ab (dict): mapping dictionary of antibody names
            Ex: {"CD4": "AB_XXXXXXX",
                 "CD5": "AB_XXXXXXX" 
                 ...}
    """
    target_ab = {}
    for pair in antibody_pairs:
        target_ab[pair[0]] = pair[1]
        
    return target_ab

def _filter_antibodies(protein_matrix: pd.DataFrame,
                       csv_file: str) -> pd.DataFrame:
    """
    Filters ADT protein table using only antibodies found in
    a user-provided spreadsheet. Used for uploads to database

    Parameters:
        protein_matrix (pd.DataFrame): protein data
        csv_file (str): file path to provided spreadsheet

    Returns:
        filt_df (pd.DataFrame): dataframe containing rows
            that reflect antibodies listed in the spreadsheet
    """

    antibody_pairs = _read_antibodies(csv_file)
    antibodies_list = [ab[0] for ab in antibody_pairs]

    # Subset the columns in the dataframe that are in our spreadsheet
    filt_df = protein_matrix.T.loc[antibodies_list]

    return filt_df.T

def _load_labels(labels: str | pd.DataFrame) -> pd.DataFrame:
    """
    Shifts the first column (cell names) to become the index.
    Remaining columns will contain the labels and celltypes.

    Parameters:
        cell_label_df (Pandas DataFrame): cell types 
            Where: rows (Index) = cells, columns = labels, celltype

    Returns:
        cell_label_modified (Pandas DataFrame): cell types, where
            the index is the cell barcode names, and there are 
            columns that can either contain labels or celltypes, or both.
    """
    if isinstance(labels, str):
        cell_label_modified = pd.read_csv(labels, sep=",", index_col=[0])
        # cell_label_modified = labels_df.copy(deep=True)
        # # Shift the first column to be the index
        # cell_label_modified.index = cell_label_modified.iloc[:, 0]
        # # Drop first column
        # cell_label_modified.drop(columns=labels_df.columns[0],
        #                             axis=1,
        #                             inplace=True)
    else:
        cell_label_modified = labels

    return cell_label_modified

def _log_transform(d_vect: list,
                  scale: (int) = 1) -> list:
    """
    Applies log transformation to a list containing raw values

    Parameters:
        d_vect (list): raw counts from protein data
        scale (int): transformation scale value (optional)

    Returns:
        data_array: log-transformed counts
    """
    min_vect = min(d_vect)
    data_array = np.array([scale*i + 1 - (scale*min_vect) for i in d_vect])
    return np.log(data_array)

def _arcsinh_transform(d_vect: list,
                      scale: int = 1) -> list:
    """
    Applies arcsinh transformation to a list containing raw values

    Parameters:
        d_vect (list): raw counts from protein data
        scale (int): transformation scale value (optional)

    Returns:
        data_array: arcsinh-transformed counts
    """
    min_vect = min(d_vect)
    data_array = np.array([scale*i + 1 - (scale*min_vect) for i in d_vect])
    return np.arcsinh(data_array.astype(float))

def _conv_np_mode(n: float,
                  p: float) -> float:
    """
    Converts Negative Binomial n, p parameters into mode

    Parameters:
        n (float): shape parameter used in Negative Binomial
        p (float): shape parameter used in Negative Binomial

    Returns:
        mode (float): mode of mixture model
    """
    if n <= 1:
        mode = 0
    elif n > 1:
        mode = (((n - 1)*(1 - p))/(p))

    return mode

def _find_background_comp(fit_results: dict) -> int:
    """
    For an antibody, find the background component in a mixture model by
    using the smallest mean

    Parameters:
        fit_results (dict): optimization results for an antibody, containing
            three mixture models and their respective parameter results
            Example:
                {3: {},
                 2: {},
                 1: {}}

    Returns:
        background_component (int): index of the background component in
            a mixture model.
            Example:
                0 = 1st component
                1 = 2nd component
                2 = 3rd component
    """

    best_num_comp = next(iter(fit_results.items()))[0]
    mix_model = next(iter(fit_results.items()))[1]

    # Check if results is from Negative Binomial or Gaussian
    if mix_model['model'] == 'negative binomial (MLE)':
        component_means_nb = []

        n_params = mix_model['nb_n_params']
        p_params = mix_model['nb_p_params']

        # Calculate means for each individual component
        for comp in range (best_num_comp):
            mean = ss.nbinom.mean(n_params[comp], p_params[comp])
            component_means_nb.append(mean)

        # Return component with the smallest mean
        background_component = component_means_nb.index(min(component_means_nb))

    elif mix_model['model'] == 'gaussian (EM)':
        gmm_means = mix_model['gmm_means']
        background_component = gmm_means.index(min(gmm_means))

    return background_component

def _classify_cells(fit_results: dict,
                    data_vector: list,
                    bg_comp: int,
                    epsilon: float = 0.5) -> list:
    """
    Classifies cells as either background or signal for a single antibody

    Parameters:
        fit_results (dict): optimization results for an antibody, containing
            three mixture models and their respective parameter results
        data_vector (list): raw counts from protein data
        bg_comp (int): integer representing background component
        epsilon (float): adjustable value added to probabilities of signal cells

    Returns:
        classified_cells (list): list of cells, with "0" indicating background
            cell and "1" indicating signal cell
    """
    classified_cells = []

    mix_model = next(iter(fit_results.items()))[1]
    best_num_mix = next(iter(fit_results.items()))[0]

    component_list = [x for x in range(best_num_mix)] # ex: [0, 1] or [0, 1, 2]

    if best_num_mix == 1:
        classified_cells = np.zeros_like(data_vector)
        return list(classified_cells)
    
    # Convert data_vector into a numpy array of floats
    data_vector = np.array(data_vector, dtype=float)

    if mix_model['model'] == 'negative binomial (MLE)':
        n_params = mix_model['nb_n_params']
        p_params = mix_model['nb_p_params']
        nb_thetas = mix_model['nb_thetas']

        bg_n = n_params[bg_comp]
        bg_p = p_params[bg_comp]
        tempThetas = nb_thetas.copy()

        component_list.remove(bg_comp)

        if best_num_mix == 3:
            tempThetas.append(1 - nb_thetas[0] - nb_thetas[1])
            
            comp2_index = component_list.pop(0)
            comp2_theta = tempThetas[comp2_index]
            comp2_probs = comp2_theta * ss.nbinom.pmf(data_vector, n_params[comp2_index], p_params[comp2_index])

            comp3_index = component_list.pop(0)
            comp3_theta = tempThetas[comp3_index]
            comp3_probs = comp3_theta * ss.nbinom.pmf(data_vector, n_params[comp3_index], p_params[comp3_index])

        elif best_num_mix == 2:
            tempThetas.append(1 - nb_thetas[0])
            
            comp2_index = component_list.pop(0)
            comp2_theta = tempThetas[comp2_index]
            comp2_probs = comp2_theta * ss.nbinom.pmf(data_vector, n_params[comp2_index], p_params[comp2_index])

        bg_theta = tempThetas[bg_comp]
        comp1_probs = bg_theta * ss.nbinom.pmf(data_vector, bg_n, bg_p)
        comp1_mode = _conv_np_mode(bg_n, bg_p)

        max_probs = np.column_stack((comp1_probs, comp2_probs, comp3_probs)) if best_num_mix == 3 else np.column_stack((comp1_probs, comp2_probs))

        classified_cells = np.argmax(max_probs, axis=1)
        classified_cells = np.where(np.logical_or(classified_cells == 0, data_vector < comp1_mode), 0, 1)

    elif mix_model['model'] == 'gaussian (EM)':
        gmm_means = mix_model['gmm_means']
        gmm_stdevs = mix_model['gmm_stdevs']
        gmm_thetas = mix_model['gmm_thetas']

        bg_mean = gmm_means[bg_comp]
        bg_stdev = gmm_stdevs[bg_comp]
        bg_theta = gmm_thetas[bg_comp]

        component_list.remove(bg_comp)

        comp1_mode = bg_mean

        if best_num_mix == 3:
            comp2_index = component_list.pop(0)
            comp2_mean = gmm_means[comp2_index]
            comp2_stdev = gmm_stdevs[comp2_index]
            comp2_theta = gmm_thetas[comp2_index]

            comp3_index = component_list.pop(0)
            comp3_mean = gmm_means[comp3_index]
            comp3_stdev = gmm_stdevs[comp3_index]
            comp3_theta = gmm_thetas[comp3_index]

        elif best_num_mix == 2:
            comp2_index = component_list.pop(0)
            comp2_mean = gmm_means[comp2_index]
            comp2_stdev = gmm_stdevs[comp2_index]
            comp2_theta = gmm_thetas[comp2_index]

        comp1_cell_probs = bg_theta * ss.norm.pdf(data_vector, bg_mean, bg_stdev)
        comp2_cell_probs = comp2_theta * ss.norm.pdf(data_vector, comp2_mean, comp2_stdev) + (0.5 - epsilon)
        comp3_cell_probs = comp3_theta * ss.norm.pdf(data_vector, comp3_mean, comp3_stdev) + (0.5 - epsilon) if best_num_mix == 3 else None

        max_probs = np.column_stack((comp1_cell_probs, comp2_cell_probs, comp3_cell_probs)) if best_num_mix == 3 else np.column_stack((comp1_cell_probs, comp2_cell_probs))

        classified_cells = np.argmax(max_probs, axis=1)
        classified_cells = np.where(np.logical_or(classified_cells == 0, data_vector < comp1_mode), 0, 1)

    return list(classified_cells)

def _classify_cells_df(fit_all_results: dict,
                       protein_data: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters:
        fit_all_results (dict): optimization results for all antibodies
        protein_data (pd.DataFrame): count matrix containing antibodies x cells

    Returns:
        classified_df (pd.DataFrame): Pandas DataFrame containing "0" and "1",
            where "0" represents a background expression of a cell for a given
            antibody, and "1" represents a antibody-specific signal expression
    """

    # Storing antibody name with each classified vector of cells
    count_matrix_columns = list(protein_data.columns)
    count_matrix_index = list(protein_data.index)

    classified_matrix = []

    for index, ab in enumerate(protein_data):
        # Find background component
        ab_background_comp_index = _find_background_comp(fit_all_results[ab])

        # Classify cells in vector as either background or signal
        ab_classified_cells = _classify_cells(fit_all_results[ab],
                                              protein_data.loc[:, ab],
                                              ab_background_comp_index)

        classified_matrix.append(ab_classified_cells)

    # Transpose matrix:
    classified_transpose = list(map(list, zip(*classified_matrix)))

    classified_df = pd.DataFrame(classified_transpose,
                                 index=count_matrix_index,
                                 columns=count_matrix_columns)

    return classified_df

def _filter_classified_df(classified_df: pd.DataFrame,
                          sig_threshold: float = 0.85,
                          bg_threshold: float = 0.15) -> pd.DataFrame:
    """
    Filters out cells that have a total signal expression ratio greater
    than a defined threshold

    Parameters:
        classified_df (pd.DataFrame): DataFrame containing all cells and their
            classification as background or signal for a set of antibodies
        sig_expr_threshold (float): threshold for antibody expression when
                filtering cells that have a high signal expression rate
        bg_expr_threshold (float): threshold for antibody expression when
            filtering cells that have a low signal expression rate

    Returns:
        filtered_df (pd.DataFrame): DataFrame containing cells that fall below
            the defined threshold for signal expression
    """
    num_ab = len(classified_df.columns)
    # num_original_cells = len(classified_df.index)

    # We want to first filter out cells that have 0 expression automatically
    # none_filt = classified_df[(classified_df == 1).sum(axis=1) > 0]

    # # We also want to filter out cells that have 100 expression automatically
    # all_filt = none_filt[(classified_df == 1).sum(axis=1) < num_ab]

    # num_filt = num_original_cells - len(all_filt.index)
    # logging.warning(f" {num_filt} cells with 0% or 100% expression have been "
    #                 "automatically filtered out.")

    # Filter out cells that have a majority expresssing signal (user-defined)
    filtered_df = classified_df[(((classified_df == 1).sum(axis=1)
                            / num_ab)) <= sig_threshold]

    # Filter out cells that have a majority expressing background (user-defined)
    filtered_df = filtered_df[(((filtered_df == 1).sum(axis=1)
                                / num_ab) >= bg_threshold)]

    # Clarifying messages for total number of filtered cells
    additional_filt = len(classified_df.index) - len(filtered_df.index)
    logging.warning(f" {additional_filt} additional cells have been filtered "
                    f"based on {sig_threshold} sig_expr and {bg_threshold} "
                    "bg_expr thresholds.")

    # total_filt = num_filt + additional_filt
    total_filt = additional_filt
    logging.warning(f" Total cells filtered: {total_filt}")

    return filtered_df

def _filter_count_df(filtered_classified_df: pd.DataFrame,
                     protein_data: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out cells in protein_data that were filtered out from classified_df,
    since the cells present in the protein data will affect the z score
    calculations

    Parameters:
        filtered_classified_df (pd.DataFrame): DataFrame containing cells
            that fall below the defined threshold for signal expression
        protein_data (pd.DataFrame): count matrix containing antibodies x cells

    Returns:
        Filtered protein count matrix
    """
    return protein_data.loc[filtered_classified_df.index]

def _filter_cell_labels(filtered_classified_df: pd.DataFrame,
                        cell_label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters out cells from cell labels for single-cell data that were
    previously filtered out in classified_df

    Parameters:
        filtered_classified_df (pd.DataFrame): DataFrame containing cells
            that fall below the defined threshold for signal expression
        cell_label_df (pd.DataFrame): DataFrame containing
            cell labels for each cell (cell x cell label)

    Returns:
        Filtered cell label DataFrame
    """
    return cell_label_df.loc[filtered_classified_df.index]

def _z_scores(fit_results: dict,
              data_vector: list) -> list:
    """
    Calculates the z scores of all cells for an antibody using only values
    from background cells. Further filtering for only background z scores
    will be required.

    Parameters:
        fit_results (dict): optimization results for an antibody
        data_vector (list): raw protein count values for an antibody

    Returns:
        z_scores (list): all count values (both background and signal) converted
            into z scores
    """
    z_scores = []

    # Find the background component
    background_comp_index = _find_background_comp(fit_results=fit_results)

    # Classify cells as either background or signal
    classified_cells = _classify_cells(fit_results=fit_results,
                                      data_vector=data_vector,
                                      bg_comp=background_comp_index)

    # Find all background cells in data vector
    background_counts = [val
                         for val, matrix_status
                         in zip(data_vector, classified_cells)
                         if matrix_status == 0]

    # Find mean of background cells (for z score calculation)
    background_mean = np.mean(background_counts)

    # Find standard deviation of background cells (for z score calculation)
    background_stdev = np.std(background_counts)

    # Convert all counts into z scores
    for count in data_vector:
        z_score = (count - background_mean) / background_stdev
        z_scores.append(z_score)

    return z_scores

def _z_scores_df(fit_all_results: dict,
                 protein_data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the z scores for all cells for a set of antibodies. Further
    filtering for only background z scores will be required. This will have
    already excluded the cells that fell above an expression threshold.

    Parameters:
        fit_all_results (dict): optimization results for all antibodies
        protein_data (pd.DataFrame): count matrix containing antibodies x cells

    Returns:
        z_score_df (pd.DataFrame): all count values in protein_data converted
            into z scores
    """

    # Storing ab name with each classified vector of cells
    count_matrix_columns = list(protein_data.columns)
    count_matrix_index = list(protein_data.index)

    z_score_matrix = []

    for index, ab in enumerate(protein_data):
        # Classify cells in vector as either background or signal
        try:
            z_score_vector = _z_scores(fit_all_results[ab], 
                                    protein_data.loc[:, ab])
        except:
            raise Exception(f"No background cells found for {ab}. " 
                            f"Please adjust fits for {ab} with select_mixture_model().")

        z_score_matrix.append(z_score_vector)

    # Transpose matrix:
    z_score_transpose = list(map(list, zip(*z_score_matrix)))

    z_score_df = pd.DataFrame(z_score_transpose,
                              index=count_matrix_index,
                              columns=count_matrix_columns)

    return z_score_df

def _bg_z_scores_df(classified_df: pd.DataFrame,
                    z_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retrieves only the z scores for background cells

    Parameters:
        classified_df (pd.DataFrame): Pandas DataFrame containing "0" and "1",
            where "0" represents a background expression of a cell for a given
            antibody, and "1" represents a antibody-specific signal expression
        z_scores_df (pd.DataFrame): all count values converted into z scores
            using mean and standard deviation of the background cells

    Returns:
        bg_z_scores_df (pd.DataFrame): z scores that belong to cells classified
            as background. Signal cells will be presented as "NaN"
    """
    classified_copy = classified_df.copy(deep=True)

    # Create a boolean dataframe to filter out background z-scores
    classified_copy.replace({0:True, 1:False}, inplace=True)

    # Create a dataframe with z scores for background, NaN for signals
    bg_z_scores_df = z_scores_df[classified_copy]

    return bg_z_scores_df

def _z_avg_umi_sum(bg_z_score_df: pd.DataFrame,
                   rna_counts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the average of z scores for a cell and their total UMIs (for
    single-cell data)

    Parameters:
        bg_z_score_df (pd.DataFrame): z scores for background cells
        rna_counts_df (pd.DataFrame): UMI counts from RNA data (cells x genes)

    Returns:
        z_umi_df (pd.DataFrame): DataFrame containing the z score average and
            total UMIs for a given cell
    """
    z_umi_df = pd.DataFrame(columns=['z_score_avg', 'total_umi'])
    z_umi_df['z_score_avg'] = bg_z_score_df.mean(axis=1)
    z_umi_df['total_umi'] = rna_counts_df.loc[z_umi_df.index].sum(axis=1)
    # z_umi_clean = z_umi_df.dropna() # don't do it here! do it when we actually call lin_reg

    return z_umi_df

def _z_avg_umi_sum_by_type(bg_z_score_df: pd.DataFrame,
                           rna_counts_df: pd.DataFrame,
                           labels_filt_df: pd.DataFrame) -> dict:
    """
    Returns a dictionary that sorts cells by cell type along with
    their z score averages and total UMIs

    Parameters:
        bg_z_score_df (pd.DataFrame): DataFrame containing only z scores for
            background cells
        rna_counts_df (pd.DataFrame): UMI counts for each cell (RNA data)
        labels_filt_df (pd.DataFrame): cells and their cell type
    Returns:
        df_by_type_dict (dict): dictionary containing keys as cell type and
            values as a dataframe consisting of z score averages and total UMIs
            for all cells for that cell type
    """

    df_by_type = []
    df_by_type_dict = {}

    z_umi_df = pd.DataFrame(columns=['z_score_avg', 'total_umi'])
    z_umi_df['z_score_avg'] = bg_z_score_df.mean(axis=1)
    z_umi_df['total_umi'] = rna_counts_df.loc[z_umi_df.index].sum(axis=1)

    # Add the list of types into the z_umi_df
    z_umi_df['type'] = list(labels_filt_df.iloc[:, 0])

    # Filter cells that contain any NA values in their z scores average
    # z_umi_clean = z_umi_df.dropna() # don't do it here! do it when we actually call lin_reg

    # Find all cell types
    unique_cell_types = list(set(labels_filt_df.iloc[:, 0]))

    # Separate all cells out by cell type
    for cell_type in unique_cell_types:
        temp_df = pd.DataFrame(z_umi_df.loc[z_umi_df['type'] == cell_type])
        df_by_type.append(temp_df)
        df_by_type_dict[cell_type] = temp_df

    return df_by_type_dict

def _linear_reg(z_umi: pd.DataFrame) -> pd.DataFrame:
    """
    Performs linear regression using z score averages and the log values
    of the total UMIs

    Parameters:
        z_umi (pd.DataFrame): DataFrame containing z score averages and total
            UMIs of cells

    Returns:
        lin_reg_df (pd.DataFrame): DataFrame containing the predicted z scores,
            residuals, and p-values for each cell
    """

    # The final predicted values
    lin_reg_df = z_umi.copy(deep=True)
    z_umi_og_x = np.log(z_umi['total_umi'].values)
    z_umi_og_y = z_umi['z_score_avg']

    # Filter out any cells (rows) containing NAs that would have caused
    # the Linear Regression to crash.
    # This is ONLY for the purpose of creating the Linear Regression model
    # The final predicted values will still be using the ORIGINAL cells
    z_umi_clean = z_umi.dropna()

    # Log transform the umi values (x axis)
    x_val = np.log(z_umi_clean['total_umi'].values)
    y_val = z_umi_clean['z_score_avg']

    # Find p-value of beta1 (coefficient)
    X2 = sm.add_constant(x_val)
    ols = sm.OLS(y_val, X2)
    ols_fit = ols.fit()

    # Create linear model
    lm = LinearRegression()

    # Fit model
    lm.fit(X=x_val.reshape(-1, 1), y=y_val)

    # Find predicted values
    lm_predict = lm.predict(z_umi_og_x.reshape(-1, 1))

    # Calculate residuals
    lm_residuals = z_umi_og_y - lm_predict

    # Add everything to dataframe
    lin_reg_df['predicted'] = lm_predict
    lin_reg_df['residual'] = lm_residuals
    lin_reg_df['p_val'] = ols_fit.f_pvalue

    return lin_reg_df

def _linear_reg_by_type(df_by_type_dict: dict) -> dict:
    """
    Performs linear regression on cells separated by cell type

    Parameters:
        df_by_type_dict: dictionary containing keys as cell type and
            values as a dataframe consisting of z score averages and total UMIs
            for all cells for that cell type
            Example:
            {
                1: {dataframe},
                2: {dataframe} ...
            }

    Returns:
        lin_reg_by_type_dict (dict): dictionary containing keys as cell type
            and values as a dataframe consisting of results from linear
            regression (predicted, residual, p_val) for that cell type
    """

    # Perform linear regression on each df for their respective cell type
    lin_reg_by_type_dict = {}

    for cell_type, df in df_by_type_dict.items():
        # If a cell type's df is empty, ignore it (no cells for that type)
        if len(df.index) == 0:
            continue
        else:
            temp_lin_reg = _linear_reg(df)
            lin_reg_by_type_dict[cell_type] = temp_lin_reg

    return lin_reg_by_type_dict

def _get_cell_type(cell: str,
                   cell_labels_filt_df: pd.DataFrame) -> str:
    """
    Uses the filtered cell type DataFrame to find
    a cell's corresponding cell type

    Parameters:
        cell (str): name of the cell
        cell_labels_filt_df (pd.DataFrame): DataFrame of cells and their types

    Returns:
        cell_type (str): the cell type
    """

    cell_type = cell_labels_filt_df.loc[cell].values[0]

    return cell_type

def _bg_mean_std(fit_results) -> tuple:
    """
    Finds the mean and standard deviation of the fit results for an antibody

    Parameters:
        fit_results (dict): optimization results for an antibody

    Returns:
        mean (float): mean value of the background component
        std (float): standard deviation of the background component
    """

    # Find background component
    background_component = _find_background_comp(fit_results)

    # Retrieve n and p parameters for all components in mix_model
    mix_model = next(iter(fit_results.items()))[1]

    if mix_model['model'] == 'negative binomial (MLE)':

        n_params = mix_model['nb_n_params']
        p_params = mix_model['nb_p_params']

        # For the mix_model, find the background component's parameters
        bg_n = n_params[background_component]
        bg_p = p_params[background_component]

        # Calculate the mean of the background component
        mean = ss.nbinom.mean(bg_n, bg_p)

        #Calculate the standard deviation
        std = ss.nbinom.std(bg_n, bg_p)

    elif mix_model['model'] == 'gaussian (EM)':
        gmm_means = mix_model['gmm_means']
        gmm_stdevs = mix_model['gmm_stdevs']

        mean = gmm_means[background_component]
        std = gmm_stdevs[background_component]

    return mean, std

def _correlation_ab(classified_df: pd.DataFrame,
                    z_scores_df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters:
        classified_df (pd.DataFrame): DataFrame of all cells and their
            classification as background or signal for a set of antibodies
            Format: rows: cells, col: antibodies
        z_scores_df (pd.DataFrame): DataFrame of z-scores of protein counts
            Format: rows: cells, col: antibodies

    Returns:
        correlation_df (pd.DataFrame): DataFrame containing pearson's correlation
            coefficients for each pair of antibodies
    """

    classified_transpose = classified_df.copy(deep=True).T
    z_transpose = z_scores_df.copy(deep=True).T

    # Before we start, we want to remove any rows that contain inf or NaN
    z_transpose.replace([np.inf, -np.inf], np.nan, inplace=True)
    z_transpose.dropna(axis=0, how='any', inplace=True)

    # Remove same rows from classified
    classified_transpose = classified_transpose.loc[z_transpose.index]

    ab_names = list(classified_transpose.index)
    num_ab = len(ab_names)

    correlation_matrix = np.zeros([num_ab, num_ab])

    for i in range(0, num_ab):
        for j in range(i + 1, num_ab):

            ab_a_zscores = []
            ab_b_zscores = []

            ab_a = list(classified_transpose.iloc[i])
            ab_b = list(classified_transpose.iloc[j])

            ab_bool = [x == 0 and y == 0 for x, y in zip(ab_a, ab_b)]
            ab_pair_bool = list(np.where(ab_bool)[0])

            z_score_a = np.array(z_transpose.iloc[i])
            z_score_b = np.array(z_transpose.iloc[j])

            # Add all z scores that belong to True in ab_pair_bool
            ab_a_zscores.extend(z_score_a[ab_pair_bool])
            # Ignore constant arrays that break the pearson correlation
            if ab_a_zscores.count(ab_a_zscores[0]) == len(ab_a_zscores):
                break

            ab_b_zscores.extend(z_score_b[ab_pair_bool])
            if ab_b_zscores.count(ab_b_zscores[0]) == len(ab_b_zscores):
                break

            # Calculate pearson correlation from both z score lists
            correlation = ss.pearsonr(ab_a_zscores, ab_b_zscores)

            # Add value to matrix
            correlation_matrix[i][j] = correlation[0]
            correlation_matrix[j][i] = correlation[0]
            correlation_matrix[i][i] = 1
            correlation_matrix[i + 1][i + 1] = 1

    correlation_df = pd.DataFrame(correlation_matrix,
                                  index=ab_names,
                                  columns=ab_names)
    return correlation_df

def _normalize_antibody(fit_results: dict,
                        data_vector: pd.Series,
                        ab_name: str,
                        p_threshold: float = 0.05,
                        background_cell_z_score = -10,
                        classified_filt_df: pd.DataFrame = None,
                        cell_labels_filt_df: pd.DataFrame = None,
                        lin_reg_dict: dict = None,
                        lin_reg: pd.DataFrame = None) -> list:
        """
        Normalizes single cell or cytometry values for an antibody

        Parameters:
            fit_results (dict): optimization results for an antibody, containing
                three mixture models and their respective parameter results
            data_vector (pd.Series): raw ADT values from protein data
            ab_name (str): name of antibody
            p_threshold (float): threshold for p value rejection
            background_cell_z_score (int): z-score value for background cells
                when computing a z-score table for all normalized counts
            classified_filt_df (pd.DataFrame): contains classification for
                all cells except those filtered by expression level
            cell_labels_filt_df (pd.DataFrame): contains cell
                labels for all cells except those filtered by expression level
            lin_reg_dict (dict): results of linear regression separated by
                cell type
            lin_reg (pd.DataFrame): results of linear regression without
                separating by cell type

        Returns:
            normalized_z_scores (list): normalized z scores for an antibody
        """
        normalized_counts = []
        normalized_z_scores = []

        # Get background mean and standard deviation of either GMM or NB model (population statistics)
        # background_mean, background_std = _bg_mean_std(fit_results)

        # Get the classified cell vector for this antibody
        classified_cells = classified_filt_df[ab_name].values

        # Find all background cells in data vector (using sample statistics here instead, since z scores used them)
        background_counts = [val
                            for val, matrix_status
                            in zip(data_vector, classified_cells)
                            if matrix_status == 0]

        # Find mean of background cells ( for z score calculation)
        background_mean = np.mean(background_counts)

        # Find standard deviation of background cells (for z score calculation)
        background_std = np.std(background_counts)

        # Raw data vector
        for i, (cell_name, cell_count) in enumerate(data_vector.items()):
            if classified_cells[i] == 0:
                normalized_counts.append(0)

            elif classified_cells[i] == 1:

                # If dealing with cytometry data
                if (cell_labels_filt_df is None
                    and lin_reg_dict is None
                    and lin_reg is None):
                    # Apply normalization formula
                    normalized_val = cell_count - background_mean
                    normalized_counts.append(normalized_val)

                # If dealing with single cell data, without cell_labels
                elif (cell_labels_filt_df is None
                      and lin_reg_dict is None
                      and lin_reg is not None):

                    # If the p_val (lin reg) is < threshold, add to this factor
                    # Otherwise, keep the factor equal to 0
                    factor = 0

                    if lin_reg.loc[cell_name]['p_val'] < p_threshold:
                        predicted = lin_reg.loc[cell_name]['predicted']

                        if math.isnan(predicted) is False:
                            factor += predicted
                        else:
                            factor += 0

                    # Apply normalization formula
                    normalized_val = (cell_count
                                - (factor)*background_std
                                - background_mean)

                    normalized_counts.append(normalized_val)

                # If dealing with single cell data, with cell labels
                elif (cell_labels_filt_df is not None
                      and lin_reg_dict is not None
                      and lin_reg is None):

                    factor = 0

                    cell_type = _get_cell_type(cell_name, cell_labels_filt_df)

                    # Look up this cell's linear regression dataframe in the dict
                    if lin_reg_dict[cell_type].loc[cell_name]['p_val'] < p_threshold:
                        # We can regress out the predicted z score
                        predicted = lin_reg_dict[cell_type].loc[cell_name]['predicted']

                        if math.isnan(predicted) is False:
                            factor += predicted
                        else:
                            factor += 0

                        # Regress out on the residual
                        residual = lin_reg_dict[cell_type].loc[cell_name]['residual']

                        if math.isnan(residual) is False:
                            factor += residual
                        else:
                            factor += 0

                    # Apply normalization formula
                    normalized_val = (cell_count
                                - (factor)*background_std
                                - background_mean)

                    normalized_counts.append(normalized_val)

        norm_signal_counts = [x for x in normalized_counts if x != 0]

        if len(norm_signal_counts) < 2:
            normalized_z_scores = [background_cell_z_score] * len(data_vector)

            # If there are too few signal cells for this antibody, we will consider it all to be
            # background. The classification values for this antibody must also be updated
            # to be all 0s (background)
            classified_filt_df[ab_name].values[:] = 0

            return normalized_z_scores
        else:
            # For all a' values (non 0), calculate the mean and standard deviation
            norm_sig_mean = np.mean(norm_signal_counts)
            norm_sig_stdev = np.std(norm_signal_counts)

            # Find the z_scores
            for count in normalized_counts:
                if count != 0:
                    temp_z_score = (count - norm_sig_mean) / (norm_sig_stdev)
                    if temp_z_score < background_cell_z_score:
                        temp_z_score = background_cell_z_score

                        # If we are setting this to the background_cell_z_score,
                        # the classification value must also be updated for this cell & antibody
                        # to be 0 (background) instead of 1 (signal)
                        classified_filt_df.at[cell_name, ab_name] = 0

                    normalized_z_scores.append(temp_z_score)
                elif count == 0:
                    normalized_z_scores.append(background_cell_z_score)

            return normalized_z_scores

def _cumulative_dist(normalized_z_score: float) -> float:
    """
    Calculates the cumulative probability for a value
    in a normal distribution

    Parameters:
        normalized_z_score (float): z_score of a cell for a given antibody 
            retrieved from the normalized protein data

    Returns:
        cdf_value: cumulative probability from a normal distribution
    """

    # Use standard normal distribution, mean = 0, stdev = 1
    cdf_value = ss.norm.cdf(normalized_z_score, 0, 1)
    return cdf_value

def _normalize_antibodies_df(protein_cleaned_filt_df: pd.DataFrame,
                             fit_all_results: dict,
                             p_threshold: float = 0.05,
                             background_cell_z_score: int = -10,
                             classified_filt_df: pd.DataFrame = None,
                             cell_labels_filt_df: pd.DataFrame = None,
                             lin_reg_dict: dict = None,
                             lin_reg: pd.DataFrame = None,
                             cumulative: bool = False) -> pd.DataFrame:
    """
    Normalizes all antibodies in a protein dataset

    Parameters:
        protein_cleaned_filt_df (pd.DataFrame): containing all cells and
            antibodies, except those that have been filtered out based
            on level of expression
        fit_all_results (dict): list of dictionaries containing
            optimization results for each antibody
        p_threshold (float): threshold for p value rejection
        background_cell_z_score (int): z-score value for background cells
            when computing a z-score table for all normalized counts
        classified_filt_df (pd.DataFrame): contains classification for
            all cells except those filtered by expression level
        cell_labels_filt_df (pd.DataFrame): contains cell
            labels for all cells except those filtered by expression level
        lin_reg_dict (dict): results of linear regression separated by
            cell type
        lin_reg (pd.DataFrame): results of linear regression without
            separating by cell type
        cumulative (bool): flag to indicate whether to return the 
            cumulative distribution probabilities

    Returns:
        normalized_df_transpose (pd.DataFrame): DataFrame containing normalized
            z-score values for all cells in a given antibody
    """

    normalized_list = []

    for ab_name, counts in tqdm(protein_cleaned_filt_df.items(), total=len(protein_cleaned_filt_df.columns)):
        norm_ab_counts = _normalize_antibody(
                            fit_results=fit_all_results[ab_name],
                            data_vector=counts,
                            ab_name=ab_name,
                            p_threshold=p_threshold,
                            background_cell_z_score=background_cell_z_score,
                            classified_filt_df=classified_filt_df,
                            cell_labels_filt_df=cell_labels_filt_df,
                            lin_reg_dict=lin_reg_dict,
                            lin_reg=lin_reg)

        normalized_list.append(norm_ab_counts)

    normalized_df = pd.DataFrame(normalized_list,
                                index=protein_cleaned_filt_df.columns,
                                columns=protein_cleaned_filt_df.index)

    # Transpose so the correct row/column labels are put
    normalized_df_transpose = normalized_df.T

    # Generate the cumulative distribution values instead for signal cells
    if cumulative is True:
        normalized_df_transpose = normalized_df_transpose.apply(
            lambda x: x.mask(x != background_cell_z_score, _cumulative_dist))

    if background_cell_z_score < 0:
        normalized_df_transpose = normalized_df_transpose - background_cell_z_score
    else:
        normalized_df_transpose = normalized_df_transpose + background_cell_z_score

    return normalized_df_transpose

def _convert_idCL_readable(idCL:str) -> str:
    """
    Converts a cell ontology id (CL:XXXXXXX) into a readable cell type name

    Parameters:
        idCL (str): cell ontology ID

    Returns:
        cellType (str): readable cell type name
        
    """
    idCL_params = {
        'q': idCL,
        'exact': 'true',
        'ontology': 'cl',
        'fieldList': 'label',
        'rows': 1,
        'start': 0
    }
    
    res = requests.get("https://www.ebi.ac.uk/ols4/api/search", params=idCL_params)
    res_JSON = res.json()
    cellType = res_JSON['response']['docs'][0]['label']
    
    return cellType

def _ebi_idCL_map(labels_df: pd.DataFrame) -> dict:
    """
    Converts a list of cell ontology IDs into readable cell type names
    as a dictionary

    Parameters:
        labels_df (pd.DataFrame): dataframe with cell labels from singleR
    
    Returns:
        idCL_map (dict) : dictionary mapping cell ontology ID to cell type
    
    """
    idCL_map = {}

    idCLs = set(labels_df["labels"])
    
    for idCL in idCLs:
        idCL_map[idCL] = _convert_idCL_readable(idCL)
    
    return idCL_map

class ImmunoPhenoError(Exception):
    """A base class for ImmunoPheno Exceptions."""

class PlotUMAPError(ImmunoPhenoError):
    """No normalized counts are found when plotting a normalized UMAP"""

class LoadMatrixError(ImmunoPhenoError):
    """Protein or transcriptome matrix is empty"""

class AntibodyLookupError(ImmunoPhenoError):
    """Antibody in protein matrix is not found"""

class TransformTypeError(ImmunoPhenoError):
    """Transformation type is neither 'log' or 'arcsinh'"""

class TransformScaleError(ImmunoPhenoError):
    """Transformation scale is not an int"""

class PlotAntibodyFitError(ImmunoPhenoError):
    """Option to view antibody fits is not a boolean"""

class PlotPercentileError(ImmunoPhenoError):
    """Percentile value for plotting is not an int or float"""

class ExtraArgumentsError(ImmunoPhenoError):
    """Providing gaussian kwargs when using negative binomial model"""

class InvalidModelError(ImmunoPhenoError):
    """Model is neither 'nb' or 'gaussian'"""

class EmptyAntibodyFitsError(ImmunoPhenoError):
    """No antibody fits present when normalizing data"""

class IncompleteAntibodyFitsError(ImmunoPhenoError):
    """Not all antibodies have been fit when normalizing data"""

class BoundsThresholdError(ImmunoPhenoError):
    """Provided threshold value lies outside of 0 <= x <= 1"""

class BackgroundZScoreError(ImmunoPhenoError):
    """Provided default z score value is greater than 0"""

class ImmunoPhenoData:
    """A class to hold single-cell data (CITE-Seq, etc) and cytometry data.
    
    Performs fitting of gaussian/negative binomial mixture models and
    normalization to antibodies present in a protein dataset. Requires protein
    data to be supplied using the protein_matrix or scanpy field.

    Args:
        protein_matrix (str | pd.Dataframe): file path or dataframe to ADT count/protein matrix. 
            Format: Row (cells) x column (antibodies/proteins).
        gene_matrix (str | pd.DataFrame): file path or dataframe to UMI count matrix.
            Format: Row (cells) x column (genes).
        cell_labels (str | pd.DataFrame): file path or dataframe to cell type labels. 
            Format: Row (cells) x column (cell type such as Cell Ontology ID). Must contain 
            a column called "labels".
        spreadsheet (str): name of csv file containing a spreadsheet with
            information about the experiment and antibodies.
        scanpy (anndata.AnnData): scanpy AnnData object used to load in protein and gene data.
        scanpy_labels (str): location of cell labels inside a scanpy object. 
            Format: scanpy is an AnnData object containing an 'obs' field
                Ex: AnnData.obs['scanpy_labels']
    """

    def __init__(self,
                 protein_matrix: str | pd.DataFrame = None,
                 gene_matrix: str | pd.DataFrame = None,
                 cell_labels: str | pd.DataFrame = None,
                 spreadsheet: str = None,
                 scanpy: anndata.AnnData = None,
                 scanpy_labels: str = None):

        # Raw values
        self._protein_matrix = protein_matrix
        self._gene_matrix = gene_matrix
        self._spreadsheet = spreadsheet
        self._cell_labels = cell_labels
        self._scanpy = scanpy

        # Temp values
        self._temp_protein = None
        self._temp_gene = None
        self._temp_labels = None

        # Calculated values
        self._all_fits = None
        self._all_fits_dict = None
        self._cumulative = False
        self._last_normalize_params = None
        self._raw_umap = None
        self._norm_umap = None
        self._umap_kwargs = None
        self._normalized_counts_df = None
        self._classified_filt_df = None
        self._cell_labels_filt_df = None
        self._linear_reg_df = None
        self._z_scores_df = None
        self._singleR_rna = None
        self._ab_ids_dict = None

        # Used when sending data to the server for running STvEA
        self._background_cell_z_score = -10
        self._stvea_correction_value = 0

        # Dataframes from run_stvea/run_dt in the ImmunoPhenoDB_Connect class
        self.distance_ratios = None
        self.cell_type_prob = None
        self.entropies = None
        self.dt_used = False

        # If loading in a scanpy object
        if scanpy is not None:
            # Extract and load protein data
            protein_anndata = scanpy[:, scanpy.var["feature_types"] == "Antibody Capture"].copy()
            protein_df = protein_anndata.to_df(layer="counts")
            self._protein_matrix = protein_df.copy(deep=True)
            self._temp_protein = self._protein_matrix.copy(deep=True)

            # Extract and load rna/gene data
            rna_anndata = scanpy[:, scanpy.var["feature_types"] == "Gene Expression"].copy()
            gene_df = rna_anndata.to_df(layer="counts")
            self._gene_matrix = gene_df.copy(deep=True)
            self._temp_gene = self._gene_matrix.copy(deep=True)

            # Filter out rna based on genes used for SingleR
            self._singleR_rna = _singleR_rna(self._gene_matrix)

            # Use scanpy cell labels if present
            if scanpy_labels is not None:
                try:
                    labels = scanpy.obs[scanpy_labels]
                    # Load these labels into the class. Create a dataframe that has "labels" and "celltype"
                    singleR_labels = pd.DataFrame(labels)
                    original_column_name = singleR_labels.columns[0]
                    singleR_labels['celltype'] = singleR_labels[original_column_name]
                    singleR_labels.rename(columns={original_column_name: 'labels'}, inplace=True)

                    self._cell_labels = singleR_labels
                    self._cell_labels_filt_df = singleR_labels
                    self._temp_labels = self._cell_labels.copy(deep=True)
                except:
                    raise Exception("Field not found in scanpy object")

        if protein_matrix is None and scanpy is None:
            raise LoadMatrixError("protein_matrix file path or dataframe must be provided")

        # if (protein_matrix is not None and
        #     cell_labels is not None and
        #     gene_matrix is None and
        #     scanpy is None):
        #     raise LoadMatrixError("gene_matrix file path or dataframe must be present along with "
        #                           "cell_labels")

        # Single cell
        if self._protein_matrix is not None and self._gene_matrix is not None and scanpy is None:
            self._protein_matrix = _load_adt(self._protein_matrix) # assume user provides cells as rows, ab as col
            self._temp_protein = self._protein_matrix.copy(deep=True)

            self._gene_matrix = _load_rna(self._gene_matrix) # assume user provides cells as rows, genes as col
            self._singleR_rna = _singleR_rna(self._gene_matrix)
            self._temp_gene = self._gene_matrix.copy(deep=True)

        # Cytometry
        elif self._protein_matrix is not None and self._gene_matrix is None and scanpy is None:
            self._protein_matrix = _load_adt(self._protein_matrix)  # assume user provides cells as rows, ab as col
            self._temp_protein = self._protein_matrix.copy(deep=True)
            self._gene_matrix = None

        # If dealing with single cell data with provided cell labels
        if self._cell_labels is not None:
            self._cell_labels = _load_labels(self._cell_labels) # assume user provides cells as rows, label as col
            self._cell_labels_filt_df = self._cell_labels.copy(deep=True) # if loading in labels intiailly, also place them in the norm labels
            self._temp_labels = self._cell_labels.copy(deep=True)
        else:
            cell_labels = None

        # If filtering antibodies using a provided spreadsheet for database uploads
        if spreadsheet is not None:
            self._protein_matrix = _filter_antibodies(self._protein_matrix, spreadsheet)
            self._temp_protein = self._protein_matrix.copy(deep=True)

            # Also create a dictionary of antibodies with their IDs for name conversion
            self._ab_ids_dict = _target_ab_dict(_read_antibodies(spreadsheet))

    def __getitem__(self, index: pd.Index | list):
        """Allows instances of ImmunoPhenoData to use the indexing operator.

        Args:
            index (pd.Index | list): list or pandas index of cell names. This will return
                a new ImmunoPhenoData object containing only those cell names in 
                all dataframes of the object. 

        Returns:
            ImmunoPhenoData: ImmunoPhenoData object with modified dataframes 
            based on provided rows/cells names.
        """
        if isinstance(index, list):
            index = pd.Index(index)

        new_instance = ImmunoPhenoData(self._protein_matrix)  # Create a new instance of the class
        
        for attr_name, attr_value in self.__dict__.items():
            if isinstance(attr_value, pd.DataFrame):
                setattr(new_instance, attr_name, attr_value.loc[index] if attr_value is not None else None)
            elif isinstance(attr_value, anndata.AnnData):
                obs_index = [i for i, obs in enumerate(attr_value.obs_names) if obs in index]
                setattr(new_instance, attr_name, attr_value[obs_index] if attr_value is not None else None)
            else:
                setattr(new_instance, attr_name, attr_value)
        
        return new_instance

    @property
    def protein(self) -> pd.DataFrame:
        """Get or set the current protein dataframe in the object.
        
        Setting a new protein dataframe requires the format to have rows (cells) and
        columns (proteins/antibodies). 

        Returns:
            pd.DataFrame: Dataframe containing protein data.
        """
        return self._protein_matrix

    @protein.setter
    def protein(self, value: pd.DataFrame) -> None:
        self._protein_matrix = value

    @property
    def rna(self) -> pd.DataFrame:
        """Get or set the current gene/rna dataframe in the object.
        
        Setting a new RNA dataframe requires the format to have rows (cells) and
        columns (genes).

        Returns:
            pd.DataFrame: Dataframe containing RNA data.
        """
        return self._gene_matrix

    @rna.setter
    def rna(self, value: pd.DataFrame) -> None:
        self._gene_matrix = value
    
    @property
    def fits(self) -> dict:
        """Get the mixture model fits for each antibody in the protein dataframe.

        Each mixture model fit will be stored in a dictionary, where the key
        is the name of the antibody. 

        Returns:
            dict: Key-value pairs represent an antibody name with a
            nested dictionary containing the respective mixture model fits. 
            Fits are ranked by the lowest AIC.
        """
        return self._all_fits_dict

    @property
    def normalized_counts(self) -> pd.DataFrame:
        """Get the normalized protein dataframe in the object.

        This dataframe will only be present if normalize_all_antibodies 
        has been called. The format will have rows (cells) and columns (proteins/antibodies).
        Note that some rows may be missing/filtered out from the normalization step.

        Returns:
            pd.DataFrame: Normalized protein counts for each antibody. 
        """
        return self._normalized_counts_df
    
    @property
    def labels(self) -> pd.DataFrame:
        """Get or set the current cell labels for all normalized cells in the object.

        This dataframe will contain rows (cells) and two columns: "labels" and "celltypes". 
        All values in the "labels" column will follow the EMBL-EBI Cell Ontology ID format.
        A common name for each value in "labels" will be in the "celltypes" column.

        Setting a new cell labels table will only update rows (cells) that are shared 
        between the existing and new table. 

        Returns:
            pd.DataFrame: Dataframe containing two columns: "labels" and "celltypes". 
        """
        return self._cell_labels_filt_df
    
    @labels.setter
    def labels(self, value: pd.DataFrame) -> None:
        self._cell_labels_filt_df = value #  Change the norm_cell_labels
        # If the cells in 'value' are found in the original table, update those rows too
        common_indices = self._cell_labels.index.intersection(self._cell_labels_filt_df.index)
        # Check for missing rows in new table that should be updated in original labels
        missing_indices = self._cell_labels_filt_df.index.difference(self._cell_labels.index)

        if not missing_indices.empty:
            print(f"Warning: The following rows were not found in the original protein dataset and will be ignored: {missing_indices.tolist()}")

        # Check for indices in raw_cell_labels that will be updated
        if not common_indices.empty:
            # Reset the UMAPs
            self._raw_umap = None
            self._norm_umap = None
            try:
                temp_norm_labels = (self._cell_labels_filt_df.loc[common_indices]).copy(deep=True)
                self._cell_labels = temp_norm_labels 
                # Update the rows in the raw_cell_labels to reflect the annotations in the norm_cell_labels
                #self._cell_labels.loc[common_indices, ['labels', 'celltype']] = self._cell_labels_filt_df.loc[common_indices, ['labels', 'celltype']]
            except Exception as e:
                print(f"An error occurred during the update: {e}")
        else:
            print("No common rows found between old and new labels. No updates will be made to the old labels.")

    def convert_labels(self) -> None:
        """Convert all cell ontology IDs to a common name.

        Requires all values in the "labels" column of the cell labels dataframe to
        follow the cell ontology format of CL:0000000 or CL_0000000.

        Returns:
            None. Modifies the cell labels dataframe in-place.
        """
        # First, check that the raw cell types table exists
        if self._cell_labels is not None and isinstance(self._cell_labels, pd.DataFrame):
            # Check if the "labels" column exists
            if "labels" in self._cell_labels:
                # Create mapping dictionary using values in the "labels" field
                labels_map = _ebi_idCL_map(self._cell_labels)

                # Map all values from dictionary back onto the "celltype" field
                # temp_df = self._cell_labels.copy(deep=True)
                self._cell_labels['celltype'] = self._cell_labels['labels'].map(labels_map)

                # Ensure all "labels" follow the format of "CL:XXXXXXX"
                self._cell_labels["labels"] = self._cell_labels["labels"].str.replace(r'^CL_([0-9]+)$', r'CL:\1', regex=True)
                
                # Set new table
                # self._cell_labels = temp_df

                # Check if normalized cell types exist. If so, repeat above
                if self.labels is not None and isinstance(self.labels, pd.DataFrame):
                    # Need to make a new labels map
                    # temp_norm_df = self._cell_labels_filt_df.copy(deep=True)
                    norm_labels_map = _ebi_idCL_map(self._cell_labels_filt_df)

                    # norm_temp_df = self.labels.copy(deep=True)
                    self._cell_labels_filt_df['celltype'] = self._cell_labels_filt_df['labels'].map(norm_labels_map)

                    # Ensure all "labels" follow the format of "CL:XXXXXXX"
                    self._cell_labels_filt_df["labels"] = self._cell_labels_filt_df["labels"].str.replace(r'^CL_([0-9]+)$', r'CL:\1', regex=True)

                    # Set new table
                    # self._cell_labels_filt_df = norm_temp_df
            else:
                raise Exception("Table does not contain 'labels' column")
        else:
            raise Exception("No cell labels found. Please provide a table with a 'labels' column.")

    def remove_antibody(self, antibody: str) -> None:
        """Removes an antibody from all protein data and mixture model fits.

        Removes all values for an antibody from all protein dataframes in-place. If
        fit_antibody or fit_all_antibodies has been called, it will also remove 
        the mixture model fits for that antibody.

        Args:
            antibody (str): name of antibody to be removed.

        Returns:
            None. Modifies all protein dataframes and fits data in-place.
        """
        if not isinstance(antibody, str):
            raise AntibodyLookupError("Antibody must be a string")
        
        column_found = False

        # Iterate through all attributes of the class
        for attr_name, attr_value in self.__dict__.items():
            # Check if the attribute is a DataFrame
            if isinstance(attr_value, pd.DataFrame):
                # Try to drop the column if it exists
                if antibody in attr_value.columns:
                    attr_value_copy = attr_value.copy()  # Create a copy to avoid SettingWithCopyWarning
                    attr_value_copy.drop(antibody, axis=1, inplace=True)
                    setattr(self, attr_name, attr_value_copy)
                    column_found = True

        if not column_found:
            raise AntibodyLookupError(f"'{antibody}' not found in protein data.")
        else:
            print(f"Removed {antibody} from object.")

        # Reset the regular and normalized UMAPs after removing an antibody from the protein dataset
        self._raw_umap = None
        self._norm_umap = None

        # CHECK: Does this antibody have a fit?
        if self._all_fits_dict != None and antibody in self._all_fits_dict:
            self._all_fits_dict.pop(antibody)
            print(f"Removed {antibody} fits.")

    def select_mixture_model(self,
                             antibody: str,
                             mixture: int) -> None:
        """Overrides the best mixture model fit for an antibody.

        Args:
            antibody (str): name of antibody to modify best mixture model fit.
            mixture (int): preferred number of mixture components to override a fit.

        Returns:
            None. Modifies mixture model order in-place.
        """
        # CHECK: is mixture between 1 and 3
        if (not 1 <= mixture <= 3):
            raise BoundsThresholdError("Number for Mixture Model must lie between 1 and 3 (inclusive).")

        # CHECK: Does this antibody have a fit?
        if self._all_fits_dict != None and antibody in self._all_fits_dict:

            # Find current ordering of mixture models for this antibody
            # We know the element at index 0 is by default the "best" (sorted by lowest AIC)
            mix_order_list = list(self._all_fits_dict[antibody].keys())

            # Find the index of the element we CHOOSE to be the best
            choice_index = mix_order_list.index(mixture)

            # SWAP the ordering of these two elements in the list
            mix_order_list[0], mix_order_list[choice_index] = mix_order_list[choice_index], mix_order_list[0]

            # With this new list ordering, re-create the dictionary
            reordered_dict = {k: self._all_fits_dict[antibody][k] for k in mix_order_list}

            # Re-assign this dictionary to this antibody key
            self._all_fits_dict[antibody] = reordered_dict
        else:
            # Else, we cannot find the antibody's fits
            raise AntibodyLookupError(f"{antibody} fits cannot be found.")

    def fit_antibody(self,
                     input: list | str,
                     ab_name: str = None,
                     transform_type: str = None,
                     transform_scale: int = 1,
                     model: str = 'gaussian',
                     plot: bool = False,
                     **kwargs) -> dict:
        """Fits a mixture model to an antibody and returns its optimal parameters.

        This function can be called to either initially fit a single antibody
        with a mixture model or replace an existing fit. This function can be called
        after fit_all_antibodies has been called to modify individual fits.

        Args:
            input (list | str): raw values from protein data or antibody name.
            ab_name (str, optional): name of antibody. Ignore if calling 
                fit_antibody by supplying the antibody name in the "input" parameter.
            transform_type (str): type of transformation. "log" or "arcsinh".
            transform_scale (int): multiplier applied during transformation.
            model (str): type of model to fit. "gaussian" or "nb".
            plot (bool): option to plot each model.
            **kwargs: initial arguments for sklearn's GaussianMixture (optional).

        Returns:
            dict: Results from optimization as either gauss_params/nb_params.
        """

        # Checking parameters
        if model != 'nb' and model != 'gaussian':
            raise InvalidModelError(("Invalid model. Please choose 'gaussian' or 'nb'. "
                                    "Default: 'gaussian'."))

        if model == 'nb' and len(kwargs) > 0:
            raise ExtraArgumentsError("additional kwargs can only be used for 'gaussian'.")

        if transform_scale != 1 and transform_type is None:
            raise TransformTypeError("transform_type must be chosen to use "
                                  "transform_scale. choose 'log' or 'arcsinh'.")

        # if isinstance(transform_scale, int) == False:
        #     raise TransformScaleError("'transform_scale' must be an integer value.")

        if isinstance(plot, bool) == False:
            raise PlotAntibodyFitError("'plot' must be a boolean value.")

        # Check if all_fits_dict exists
        if self._all_fits_dict is None:
            self._all_fits_dict = {}
            for ab in list(self._protein_matrix.columns):
                self._all_fits_dict[ab] = None

        # Indicate whether this function call is for individual fitting
        individual = False

        # Fitting a single antibody using its name
        if isinstance(input, str):
            # Use the original protein counts prior to applying any transformations
            try:
                data_vector = list(self._temp_protein.loc[:, input].values)
                # Also set ab_name to input, since input is the string of the antibody
                ab_name = input
                individual = True

                if transform_type is None:
                    # If no transform type, reset data back to normal
                    self.protein.loc[:, ab_name] = self._temp_protein.loc[:, ab_name]
                    data_vector = list(self._temp_protein.loc[:, ab_name])

            except:
                raise AntibodyLookupError(f"'{input}' not found in protein data.")
        # Using the array of antibody counts itself (when fitting all antibodies at once)
        else:
            data_vector = input

        if transform_type is not None:
            if transform_type == 'log':
                data_vector = _log_transform(d_vect=data_vector,
                                            scale=transform_scale)
                self.protein.loc[:, ab_name] = data_vector

            elif transform_type == 'arcsinh':
                data_vector = _arcsinh_transform(d_vect=data_vector,
                                                scale=transform_scale)
                self.protein.loc[:, ab_name] = data_vector

            else:
                raise TransformTypeError(("Invalid transformation type. " 
                                          "Please choose 'log' or 'arcsinh'. "
                                          "Default: None."))
        elif transform_type is None:
            # If no transform type, reset data back to normal
            self.protein.loc[:, ab_name] = self._temp_protein.loc[:, ab_name]
            data_vector = list(self._temp_protein.loc[:, ab_name])

        if model == 'gaussian':
            gauss_params = _gmm_results(counts=data_vector,
                                        ab_name=ab_name,
                                        plot=plot,
                                        **kwargs)

            # Check if a params already exists in dict
            if individual:
                # Add or replace the existing fit so far
                self._all_fits_dict[input] = gauss_params

                # Update all of the fits to self._all_fits
                # while filtering out None (for antibodies without fits yet)
                self._all_fits = list(filter(None, self._all_fits_dict.values()))

            return gauss_params

        elif model == 'nb':
            nb_params = _nb_mle_results(counts=data_vector,
                                        ab_name=ab_name,
                                        plot=plot)

            if individual:
                self._all_fits_dict[input] = nb_params
                self._all_fits = list(filter(None, self._all_fits_dict.values()))

            return nb_params

    def fit_all_antibodies(self,
                           transform_type: str = None,
                           transform_scale: int = 1,
                           model: str = 'gaussian',
                           plot: bool = False,
                           **kwargs) -> None:
        """Fits all antibodies with a Gaussian or Negative Binomial mixture model.

        Fits a Gaussian or Negative Binomial mixture model to all antibodies
        in the protein dataset. After all antibodies are fit, the output will 
        display the number of each mixture model fit in the dataset. This includes
        the names of the antibodies that were fit with a single component model.
        
        Args:
            transform_type (str): type of transformation. "log" or "arcsinh".
            transform_scale (int): multiplier applied during transformation.
            model (str): type of model to fit. "gaussian" or "nb".
            plot (bool): option to plot each model.
            **kwargs: initial arguments for sklearn's GaussianMixture (optional).
        
        Returns:
            None. Results will be stored in the class. This is accessible using 
            the "fits" property.
        """

        fit_all_results = []

        for ab in tqdm(self._protein_matrix, total=len(self._protein_matrix.columns)):
            # Use original antibody counts each time to avoid compounding transformations
            fits = self.fit_antibody(input=self._temp_protein.loc[:, ab],
                                    ab_name=ab,
                                    transform_type=transform_type,
                                    transform_scale=transform_scale,
                                    model=model,
                                    plot=plot,
                                    **kwargs)
            self._all_fits_dict[ab] = fits
            fit_all_results.append(fits)
        
        number_3_component = 0
        number_2_component = 0
        all_1_component = []

        for ab_name, fit_results in self._all_fits_dict.items():
            num_components = next(iter(fit_results))
            if num_components == 3:
                number_3_component += 1
            elif num_components == 2:
                number_2_component += 1
            elif num_components == 1:
                all_1_component.append(ab_name)
                
        print("Number of 3 component models:", number_3_component)
        print("Number of 2 component models:", number_2_component)
        print("Number of 1 component models:", len(all_1_component))

        if len(all_1_component) > 0:
            print("Antibodies of 1 component models:")
            for background_ab in all_1_component:
                print(background_ab)

        # Store in class
        self._all_fits = fit_all_results

    def normalize_all_antibodies(self,
                                 p_threshold: float = 0.05,
                                 sig_expr_threshold: float = 0.85,
                                 bg_expr_threshold: float = 0.15,
                                 bg_cell_z_score: int = 10,
                                 cumulative: bool = False) -> pd.DataFrame:
        """Normalizes all antibodies in the protein data.

        The normalization step uses the fits from the mixture model to remove 
        background noise from the overall signal expression of an antibody. This will take into
        account non-specific antibody binding if RNA data is present. If RNA data 
        is present, the effects of cell size on the background noise will be regressed out
        for cells not expressing the antibody. Likewise, if cell labels are provided, 
        the effects of cell types on the background noise for these cells will also be regressed out. 
        These effects are determined by performing a linear regression using the 
        total number of mRNA UMI counts as a proxy for cell size.

        Args:
            p_threshold (float): level of significance for testing the association
                between cell size/type and background noise from linear regression. If 
                the p-value is smaller than the threshold, these factors are regressed out.
            sig_expr_threshold (float): cells with a percentage of expressed proteins above
                this threshold are filtered out.
            bg_expr_threshold (float): cells with a percentage of expressed proteins below
                this threshold are filtered out.
            bg_cell_z_score (int): The number of standard deviations of average protein expression
                to separate cells that express an antibody from cells that do not express an antibody.
                A larger value will result in more discrete clusters in the normalized 
                protein expression space.
            cumulative (bool): flag to indicate whether to return the 
                cumulative distribution probabilities.
        
        Returns:
            None. Results will be stored in the class. This is accessible using 
            the "normalized_counts" property.
        """

        # Check if parameters have changed
        if (p_threshold, sig_expr_threshold, 
            bg_expr_threshold, bg_cell_z_score, cumulative) != self._last_normalize_params:
            # If so, reset UMAP stored in class
            self._norm_umap = None
            # Update the parameters
            self._last_normalized_params = (p_threshold, sig_expr_threshold, 
                                            bg_expr_threshold, bg_cell_z_score, cumulative)
        
        bg_cell_z_score = -bg_cell_z_score
        if self._all_fits is None:
            raise EmptyAntibodyFitsError("No fits found for each antibody. Please "
                                         "call fit_all_antibodies() or fit_antibody() first.")

        if None in self._all_fits_dict.values():
            raise IncompleteAntibodyFitsError("All antibodies must be fit before normalizing. "
                                              "call fit_all_antibodies() or fit_antibody() for "
                                              "each antibody.")

        all_fits = self._all_fits_dict

        if (not 0 <= p_threshold <= 1 or
            not 0 <= sig_expr_threshold <= 1 or
            not 0 <= bg_expr_threshold <= 1):
            raise BoundsThresholdError("threshold must lie between 0 and 1 (inclusive)")

        # if not bg_cell_z_score < 0:
        #     raise BackgroundZScoreError("bg_cell_z_score must be less than 0")

        warnings.filterwarnings('ignore')
        # Classify all cells as either background or signal
        classified_cells = _classify_cells_df(all_fits, self._protein_matrix)

        # Filter out cells that have a high signal: background ratio (default: 1.0)
        classified_cells_filt = _filter_classified_df(classified_cells,
                                                    sig_threshold=sig_expr_threshold,
                                                    bg_threshold=bg_expr_threshold)
        self._classified_filt_df = classified_cells_filt

        # Filter the same cells from the protein data
        protein_cleaned_filt = _filter_count_df(classified_cells_filt,
                                                self._protein_matrix)

        # Filter from cell labels if dealing with single cell data
        if self._cell_labels is not None and self._cell_labels_filt_df is not None:
            # Update the raw cell labels to only contain labels/cells from the norm labels
            common_indices = self._cell_labels.index.intersection(self._cell_labels_filt_df.index)
            # Check for indices in raw_cell_labels that will be updated
            if not common_indices.empty:
                # Reset the UMAPs
                self._raw_umap = None
                self._norm_umap = None
                self._cell_labels.loc[common_indices] = self._cell_labels_filt_df.loc[common_indices]
                
            cell_labels_filt = _filter_cell_labels(classified_cells_filt,
                                                        self._cell_labels)
            
            self._cell_labels_filt_df = cell_labels_filt # this will replace the norm label field directly

        # Calculate z scores for all values
        z_scores = _z_scores_df(all_fits, protein_cleaned_filt)
        self._z_scores_df = z_scores

        # Extract z scores for background cells
        background_z_scores = _bg_z_scores_df(classified_cells_filt, z_scores)

        # Set cumulative flag
        self._cumulative = cumulative

        # Set the background cell z score to the class attribute (for STvEA)
        self._background_cell_z_score = bg_cell_z_score

        # The server currently uses +10 adjustment to all background cells
        # Calculate the additional adjustment from the user-provided background cell z score
        # Example: bg_cell_z_score: -10, stvea_correction_value: 0
        # Example: bg_cell_z_score: -3, stvea_correction_value: 7
        self._stvea_correction_value = 10 + bg_cell_z_score

        # If dealing with single cell data with cell_labels,
        # Run linear regression to regress out z scores based on size and cell type
        # if self._gene_matrix is not None and self._cell_labels is not None:
        #     df_by_type = _z_avg_umi_sum_by_type(background_z_scores,
        #                                         self._gene_matrix,
        #                                         self._cell_labels_filt_df)

        #     lin_reg_type = _linear_reg_by_type(df_by_type)
        #     self._linear_reg_df = lin_reg_type

        #     # Normalize all protein values
        #     normalized_df = _normalize_antibodies_df(
        #                             protein_cleaned_filt_df=protein_cleaned_filt,
        #                             fit_all_results=all_fits,
        #                             p_threshold=p_threshold,
        #                             background_cell_z_score=bg_cell_z_score,
        #                             classified_filt_df=classified_cells_filt,
        #                             cell_labels_filt_df=self._cell_labels_filt_df,
        #                             lin_reg_dict=lin_reg_type,
        #                             cumulative=cumulative)

        # # If dealing with single cell data WITHOUT cell labels:
        # # Run linear regression to regress out only size
        # elif self._gene_matrix is not None and self._cell_labels is None:
        #     z_umi = _z_avg_umi_sum(background_z_scores, self._gene_matrix)
        #     lin_reg = _linear_reg(z_umi)
        #     self._linear_reg_df = lin_reg

        #     # Normalize all protein values
        #     normalized_df = _normalize_antibodies_df(
        #                             protein_cleaned_filt_df=protein_cleaned_filt,
        #                             fit_all_results=all_fits,
        #                             p_threshold=p_threshold,
        #                             background_cell_z_score=bg_cell_z_score,
        #                             classified_filt_df=classified_cells_filt,
        #                             lin_reg=lin_reg,
        #                             cumulative=cumulative)

        # # Else, normalize values for cytometry data
        # else:
        
        # Normalize all values in the protein matrix
        normalized_df = _normalize_antibodies_df(
                                protein_cleaned_filt_df=protein_cleaned_filt,
                                fit_all_results=all_fits,
                                p_threshold=p_threshold,
                                background_cell_z_score=bg_cell_z_score,
                                classified_filt_df=classified_cells_filt,
                                cumulative=cumulative)
        
        self._normalized_counts_df = normalized_df
        