import numpy as np
import scipy
import scipy.stats as ss
import statsmodels.api as sm
import pandas as pd
import statistics
import warnings
import logging
import csv
import multiprocessing
import multiprocess
import anndata
from importlib.resources import files
from tqdm.autonotebook import tqdm

from .models import _gmm_results, _nb_mle_results
from sklearn.linear_model import LinearRegression

def _clean_adt(protein_filepath: str) -> pd.DataFrame:
    """
    Transposes the protein DataFrame

    Parameters:
      protein_df (Pandas DataFrame): protein data, rows = proteins, cols = cells

    Returns:
      protein_df_copy (Pandas DataFrame): transposed df
    """

    protein_df = pd.read_csv(protein_filepath, sep=",", index_col=[0]).T

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
        row_indices = set(range(chunk_range[0], chunk_range[1]))
        
    with open(file_path, "r") as f:
        reader = csv.reader(f)
        
        if chunk_range is not None:
            for i, line in enumerate(reader):
                if i in row_indices:   
                    yield (line)
        else:
            for row in reader: 
                yield row

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

def _gene_name_generator(file_path: str):
    """
    Reads in an RNA csv file and produces a generator containing
    gene names for each cell as a list

    Parameters:
        file_path (str): file path to csv file
    
    Returns:
        row (generator object): generator containing all genes
            for a single cell. Stored as a list
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
    
    # Create a generator to get all the column names
    rna_data = _read_csv(gene_filepath)
    cell_columns = [next(rna_data, None) for _ in range(1)][0][1:]
    
    # Create a generator to get all gene names
    gene_rows = _gene_name_generator(gene_filepath)
    list_gene_rows = list(gene_rows)
    
    # Default chunk size will be num genes // 2
    chunk_size = len(list_gene_rows) // 2
    
    # Default partition size will be chunk_size // 3
    partition_size = chunk_size // 3

    # Generate all the ranges for genes in the dataset
    gene_ranges = _clean_chunks(_generate_range(len(list_gene_rows) + 1, 
                                               start=0, 
                                               interval_size=chunk_size))
    
    dataframe_chunks = []
    
    def process_section(chunk):
        umi_chunk = _umi_generator(gene_filepath, chunk)
        umi_chunk_df = pd.DataFrame.sparse.from_spmatrix(scipy.sparse.csr_matrix(list(umi_chunk)))
        return umi_chunk_df.T # less expensive to transpose here compare to the end
        
    for chunk in gene_ranges:
        partition_ranges = _clean_chunks(_generate_range(chunk[1], 
                                                       interval_size=partition_size, 
                                                       start=chunk[0]))
        # Create a multiprocessing pool
        with multiprocess.Pool(processes=multiprocessing.cpu_count()) as pool:
            results = pool.map(process_section, partition_ranges)

        for rna_df in results:
            dataframe_chunks.append(rna_df)
    
    # Combine all DataFrames
    results_df = pd.concat(dataframe_chunks, ignore_index=True, axis=1)
    
    # Add in Index and Columns
    cell_index = pd.Index(cell_columns)
    results_df.set_index(cell_index, inplace=True)
    results_df.columns = list_gene_rows
    
    return results_df

def _clean_rna(gene_filepath: str) -> pd.DataFrame:
    """
    Loads in RNA data from a CSV to a pandas DataFrame
    
    Parameters:
        gene_filepath (str): csv file path with rows (genes) x columns (cells)
    
    Returns:
        rna_df (pd.DataFrame): RNA dataframe with rows (cells) x columns (genes)
    """
    
    # If the number of cells is <= 20k, we can use pandas directly to load
    # Create a generator to get all the column names
    rna_data = _read_csv(gene_filepath)
    cell_columns = [next(rna_data, None) for _ in range(1)][0][1:]
    
    if len(cell_columns) <= 20000:
        rna_df = pd.read_csv(gene_filepath, sep=",", index_col=[0]).T
    
    # Otherwise, load in parallel
    else:
        rna_df = _load_rna_parallel(gene_filepath)
    
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
        antibodies (list): list of antibody ids (ex: AB_XXXXXXX)
    """
    antibodies = []

    with open(csv_file, 'r') as csv_file:
        lines = csv_file.readlines()
        num_lines = len(lines)
        ab_index = (lines.index('Antibody table:,\n') + 1)
        
        while ab_index < num_lines:
            ab = lines[ab_index].strip().split(",", maxsplit=1)
            ab_name_strip = ab[0].strip()
            ab_id_strip = ab[1].strip()
            antibodies.append([ab_name_strip, ab_id_strip])
            ab_index += 1
    
    return antibodies

def _filter_antibodies(protein_matrix: pd.DataFrame,
                       csv_file: str) -> pd.DataFrame:
    """
    Filters ADT protein table using only antibodies found in 
    a user-provided spreadsheet
    
    Parameters:
        protein_matrix (pd.DataFrame): protein data 
        csv_file (str): file path to provided spreadsheet
    
    Returns:
        filt_df (pd.DataFrame): dataframe containing rows
            that reflect antibodies listed in the spreadsheet
    """
    
    antibody_pairs = _read_antibodies(csv_file)
    antibodies_list = [ab[0] for ab in antibody_pairs]

    # Subset the columns that are in our spreadsheet
    filt_df = protein_matrix.loc[antibodies_list]
    
    return filt_df.T

def _clean_labels(cell_label_df: pd.DataFrame) -> pd.DataFrame:
    """
    Shifts the first column (cell names) to become the index.
    Remaining column will contain the cell types.

    Parameters:
        cell_label_df (Pandas DataFrame): cell types, rows = cells, cols = types
    
    Returns:
        cell_label_modified (Pandas DataFrame): cell types with only one column
    """

    cell_label_modified = cell_label_df.copy(deep=True)
    # Shift the first column to be the index
    cell_label_modified.index = cell_label_modified.iloc[:, 0]
    # Drop first column
    cell_label_modified.drop(columns=cell_label_df.columns[0],
                                axis=1,
                                inplace=True)
    
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

    if mix_model['model'] == 'negative binomial (MLE)':
        n_params = mix_model['nb_n_params']
        p_params = mix_model['nb_p_params']

        # Given background component, find its associated parameters
        bg_n = n_params[bg_comp]
        bg_p = p_params[bg_comp]

        component_list.remove(bg_comp)

        # comp1 will be the background component
        comp1_probs = ss.nbinom.pmf(range(int(max(data_vector)) + 1), 
                                    bg_n, 
                                    bg_p)

        # find the mode of the fitted background component
        comp1_mode = _conv_np_mode(bg_n, bg_p)

        if best_num_mix == 3:
            comp2_index = component_list.pop(0)
            comp2_probs = ss.nbinom.pmf(range(int(max(data_vector)) + 1),
                                        n_params[comp2_index], 
                                        p_params[comp2_index])

            comp3_index = component_list.pop(0)
            comp3_probs = ss.nbinom.pmf(range(int(max(data_vector)) + 1), 
                                        n_params[comp3_index], 
                                        p_params[comp3_index])
        elif best_num_mix == 2:
            comp2_index = component_list.pop(0)
            comp2_probs = ss.nbinom.pmf(range(int(max(data_vector)) + 1), 
                                        n_params[comp2_index], 
                                        p_params[comp2_index])
        
        # Iterate over each cell value for an antibody
        for cell in data_vector:
            # Convert to int
            cell = int(cell)

            if best_num_mix == 3:
                comp1_cell_prob = comp1_probs[cell]
                comp2_cell_prob = comp2_probs[cell] + (0.5 - epsilon)
                comp3_cell_prob = comp3_probs[cell] + (0.5 - epsilon)

                cell_probs = [comp1_cell_prob, comp2_cell_prob, comp3_cell_prob]

                if max(cell_probs) == comp1_cell_prob or cell < comp1_mode:
                    classified_cells.append(0)
                else:
                    classified_cells.append(1)
            elif best_num_mix == 2:
                comp1_cell_prob = comp1_probs[cell]
                comp2_cell_prob = comp2_probs[cell] + (0.5 - epsilon)

                cell_probs = [comp1_cell_prob, comp2_cell_prob]
                
                if max(cell_probs) == comp1_cell_prob or cell < comp1_mode:
                    classified_cells.append(0)
                else:
                    classified_cells.append(1)
            elif best_num_mix == 1:
                # if just 1 component, classify all as background
                classified_cells.append(0)

    elif mix_model['model'] == 'gaussian (EM)':
        gmm_means = mix_model['gmm_means']
        gmm_stdevs = mix_model['gmm_stdevs']

        bg_mean = gmm_means[bg_comp]
        bg_stdev = gmm_stdevs[bg_comp]

        component_list.remove(bg_comp)
        
        # mode of background will be equal to the mean
        comp1_mode = bg_mean

        if best_num_mix == 3:
            comp2_index = component_list.pop(0)
            comp2_mean = gmm_means[comp2_index]
            comp2_stdev = gmm_stdevs[comp2_index]
            
            comp3_index = component_list.pop(0)
            comp3_mean = gmm_means[comp3_index]
            comp3_stdev = gmm_stdevs[comp3_index]
        elif best_num_mix == 2:
            comp2_index = component_list.pop(0)
            comp2_mean = gmm_means[comp2_index]
            comp2_stdev = gmm_stdevs[comp2_index]
            
        # Iterate over each cell value for an antibody
        for cell in data_vector:
            # Convert to int 
            cell = int(cell)

            if best_num_mix == 3:
                comp1_cell_prob = ss.norm.pdf(cell,
                                              bg_mean,
                                              bg_stdev)
                comp2_cell_prob = ss.norm.pdf(cell,
                                              comp2_mean,
                                              comp2_stdev) + (0.5 - epsilon)
                comp3_cell_prob = ss.norm.pdf(cell,
                                              comp3_mean,
                                              comp3_stdev) + (0.5 - epsilon)

                cell_probs = [comp1_cell_prob, comp2_cell_prob, comp3_cell_prob]
                
                if max(cell_probs) == comp1_cell_prob or cell < comp1_mode:
                    classified_cells.append(0)
                else:
                    classified_cells.append(1)
            elif best_num_mix == 2:
                comp1_cell_prob = ss.norm.pdf(cell,
                                              bg_mean,
                                              bg_stdev)
                comp2_cell_prob = ss.norm.pdf(cell,
                                              comp2_mean,
                                              comp2_stdev) + (0.5 - epsilon)
                cell_probs = [comp1_cell_prob, comp2_cell_prob]
                
                if max(cell_probs) == comp1_cell_prob or cell < comp1_mode:
                    classified_cells.append(0)
                else:
                    classified_cells.append(1)
            elif best_num_mix == 1:
                classified_cells.append(0)

    return classified_cells

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
    num_original_cells = len(classified_df.index)

    # We want to first filter out cells that have 0 expression automatically
    none_filt = classified_df[(classified_df == 1).sum(axis=1) > 0]

    # We also want to filter out cells that have 100 expression automatically
    all_filt = none_filt[(classified_df == 1).sum(axis=1) < num_ab]

    num_filt = num_original_cells - len(all_filt.index)
    logging.warning(f" {num_filt} cells with 0% or 100% expression have been " 
                    "automatically filtered out.")

    # Filter out cells that have a majority expresssing signal (user-defined)
    filtered_df = all_filt[(((all_filt == 1).sum(axis=1) 
                            / num_ab)) <= sig_threshold]
    
    # Filter out cells that have a majority expressing background (user-defined)
    filtered_df = filtered_df[(((filtered_df == 1).sum(axis=1)
                                / num_ab) >= bg_threshold)]
    
    # Clarifying messages for total number of filtered cells
    additional_filt = len(all_filt.index) - len(filtered_df.index)
    logging.warning(f" {additional_filt} additional cells have been filtered "
                    f"based on {sig_threshold} sig_expr and {bg_threshold} "
                    "bg_expr thresholds.")
    
    total_filt = num_filt + additional_filt
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

    # Find mean of background cells ( for z score calculation)
    background_mean = statistics.mean(background_counts)

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
        z_score_vector = _z_scores(fit_all_results[ab], 
                                  protein_data.loc[:, ab])

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
    z_umi_clean = z_umi_df.dropna()

    return z_umi_clean

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
    
    z_umi_clean = z_umi_df.dropna()

    # Find all cell types 
    unique_cell_types = list(set(labels_filt_df.iloc[:, 0]))

    # Separate all cells out by cell type
    for cell_type in unique_cell_types:
        temp_df = pd.DataFrame(z_umi_clean.loc[z_umi_clean['type'] == cell_type])
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

    lin_reg_df = z_umi.copy(deep=True)

    # Log transform the umi values (x axis)
    x_val = np.log(z_umi['total_umi'].values)
    y_val = z_umi['z_score_avg']

    # Find p-value of beta1 (coefficient)
    X2 = sm.add_constant(x_val)
    ols = sm.OLS(y_val, X2)
    ols_fit = ols.fit()

    # Create linear model
    lm = LinearRegression()

    # Fit model
    lm.fit(X=x_val.reshape(-1, 1), y=y_val)

    # Find predicted values
    lm_predict = lm.predict(x_val.reshape(-1, 1))

    # Calculate residuals
    lm_residuals = y_val.values - lm_predict

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
        Normalizes single cell or flow cytometry values for an antibody

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

        # Get background mean and standard deviation of either GMM or NB model
        background_mean, background_std = _bg_mean_std(fit_results)

        # Get the classified cell vector for this antibody
        classified_cells = classified_filt_df[ab_name].values

        # Raw data vector
        for i, (cell_name, cell_count) in enumerate(data_vector.items()):
            if classified_cells[i] == 0:
                normalized_counts.append(0)

            elif classified_cells[i] == 1:

                # If dealing with flow cytometry data
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
                        factor += lin_reg.loc[cell_name]['predicted']
                    
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
                        factor += lin_reg_dict[cell_type].loc[cell_name]['predicted']

                    # Apply normalization formula
                    normalized_val = (cell_count 
                                - (factor)*background_std 
                                - background_mean)
                    
                    normalized_counts.append(normalized_val)
        
        norm_signal_counts = [x for x in normalized_counts if x != 0]

        if len(norm_signal_counts) < 2:
            normalized_z_scores = [background_cell_z_score] * len(data_vector)
            return normalized_z_scores
        else:
            # For all a' values (non 0), calculate the mean and standard deviation
            norm_sig_mean = statistics.mean(norm_signal_counts)
            norm_sig_stdev = statistics.stdev(norm_signal_counts)

            # Find the z_scores
            for count in normalized_counts:
                if count != 0:
                    temp_z_score = (count - norm_sig_mean) / (norm_sig_stdev)
                    if temp_z_score < background_cell_z_score:
                        temp_z_score = background_cell_z_score
                    normalized_z_scores.append(temp_z_score)
                elif count == 0:
                    normalized_z_scores.append(background_cell_z_score)
            
            return normalized_z_scores

def _normalize_antibodies_df(protein_cleaned_filt_df: pd.DataFrame, 
                             fit_all_results: dict,
                             p_threshold: float = 0.05,
                             background_cell_z_score: int = -10,
                             classified_filt_df: pd.DataFrame = None, 
                             cell_labels_filt_df: pd.DataFrame = None, 
                             lin_reg_dict: dict = None,
                             lin_reg: pd.DataFrame = None) -> pd.DataFrame:
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

    Returns:
        normalized_df_transpose (pd.DataFrame): 
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

    if background_cell_z_score < 0:
        normalized_df_transpose = normalized_df_transpose - background_cell_z_score
    else:
        normalized_df_transpose = normalized_df_transpose + background_cell_z_score

    return normalized_df_transpose

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
    """
    A class to hold single-cell data (CITE-Seq, etc) and Flow Cytometry data.
    Performs fitting of gaussian/negative binomial mixture models and
    normalization to antibodies present in a protein dataset. 

    Parameters:
        protein_matrix (str): file path to ADT count matrix with row (antibodies) x column (cells)
        gene_matrix (str): file path to UMI count matrix with row (genes) x column (cells)
        spreadsheet (str): name of csv file containing a spreadsheet with
            information about the experiment and antibodies
        scanpy (AnnData object): scanpy anndata object used to load in protein and gene data
        cell_labels (pd.DataFrame): matrix with cell x cell type (ex: Cell Ontologies)
    """
    def __init__(self, 
                 protein_matrix: str = None, 
                 gene_matrix: str = None,
                 spreadsheet: str = None,
                 scanpy: anndata.AnnData = None,
                 cell_labels: pd.DataFrame = None):
        
        # Raw values
        self._protein_matrix = protein_matrix
        self._gene_matrix = gene_matrix
        self._spreadsheet = spreadsheet
        self._cell_labels = cell_labels
        self._label_certainties = None
        self._scanpy = scanpy

        # Temp values (for resetting index)
        self._temp_protein = None
        self._temp_gene = None
        self._temp_labels = None
        self._temp_certainties = None
        
        # Calculated values
        self._all_fits = None
        self._all_fits_dict = None
        self._normalized_counts_df = None
        self._classified_filt_df = None
        self._cell_labels_filt_df = None
        self._linear_reg_df = None
        self._z_scores_df = None
        self._singleR_rna = None

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

        if protein_matrix is None and scanpy is None:
            raise LoadMatrixError("protein_matrix file path must be provided")

        if (protein_matrix is not None and 
            cell_labels is not None and 
            gene_matrix is None and
            scanpy is None):
            raise LoadMatrixError("gene_matrix file path must be present along with "
                                  "cell_labels")
        
        # Single cell
        if self._protein_matrix is not None and self._gene_matrix is not None and scanpy is None:
            self._protein_matrix = _clean_adt(self._protein_matrix)
            self._temp_protein = self._protein_matrix.copy(deep=True)

            self._gene_matrix = _clean_rna(self._gene_matrix)
            self._singleR_rna = _singleR_rna(self._gene_matrix)
            self._temp_gene = self._gene_matrix.copy(deep=True)

        # Flow
        elif self._protein_matrix is not None and self._gene_matrix is None and scanpy is None:
            self._protein_matrix = _clean_adt(self._protein_matrix)
            self._temp_protein = self._protein_matrix.copy(deep=True)
            self._gene_matrix = None

        # If dealing with single cell data
        if self._cell_labels is not None:
            self._cell_labels = _clean_labels(self._cell_labels)
            self._temp_labels = self._cell_labels.copy(deep=True)
        else:
            cell_labels = None

        # If filtering antibodies using a provided spreadsheet
        if spreadsheet is not None:
            self._protein_matrix = _filter_antibodies(self._protein_matrix.T, spreadsheet)
            self._temp_protein = self._protein_matrix.copy(deep=True)
    
    @property
    def classified_filt(self):
        return self._classified_filt_df

    @property
    def z_scores(self):
        return self._z_scores_df
    
    @property
    def normalized_counts(self):
        return self._normalized_counts_df
    
    @property
    def protein(self):
        return self._protein_matrix

    @property 
    def rna(self):
        return self._gene_matrix

    @property
    def norm_cell_labels(self):
        return self._cell_labels_filt_df
    
    @property
    def raw_cell_labels(self):
        return self._cell_labels
    
    @raw_cell_labels.setter
    def raw_cell_labels(self, value):
        self._cell_labels = value

    @property
    def label_certainties(self):
        return self._label_certainties
    
    @label_certainties.setter
    def label_certainties(self, value):
        self._label_certainties = value

    def reset_index(self):
        """
        Resets all dataframe values using original index

        """
        self._protein_matrix = self._temp_protein
        self._gene_matrix = self._temp_gene
        self._cell_labels = self._temp_labels
        self._label_certainties = self._temp_certainties

    def update_index(self,
                     index: list):
        """
        Updates the index of cells for each dataframe 
        in an ImmunoPhenoData object

        Parameters:
            index (list/pandas.core.indexes.base.Index): list of cell names
        """
        # Before updating the index, reset it back to its original
        self.reset_index()

        self._protein_matrix = self._protein_matrix.loc[index]
        self._gene_matrix = self._gene_matrix.loc[index]
        self._cell_labels = self._cell_labels.loc[index]
        self._label_certainties = self._label_certainties.loc[index]

    def remove_antibody(self,
                        antibody: str):
        """
        Removes an antibody from protein data and mixture model fits (if present)

        Parameters:
            antibody (str): name of antibody to be removed
        
        """
        # CHECK: Does this antibody exist in the protein data?
        if isinstance(antibody, str):
            try:
                # Drop column from protein data
                self._protein_matrix.drop(antibody, axis=1, inplace=True)
                print(f"Removed {antibody} from protein data.")  
            except:
                raise AntibodyLookupError(f"'{antibody}' not found in protein data.")
        else:
            raise AntibodyLookupError("Antibody must be a string")  

        # CHECK: Does this antibody have a fit?
        if self._all_fits_dict != None and antibody in self._all_fits_dict:
            self._all_fits_dict.pop(antibody)
            print(f"Removed {antibody} fits.")
    
    def select_mixture_model(self,
                             antibody: str,
                             mixture: int):
        """
        Overrides the best mixture model fit for an antibody

        Parameters:
            antibody (str): name of antibody to modify best mixture model fit
            mixture (int): number of mixture components for a given fit to override

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
                     input: list,
                     ab_name: str = None,
                     transform_type: str = None,
                     transform_scale: int = 1,
                     model: str = 'gaussian',
                     plot: bool = False,
                     **kwargs) -> dict:
        """
        Fits a mixture model to an antibody and returns its optimal parameters
        Can be called after fit_all_antibodies to replace a fit for a particular
        antibody or to fit an antibody one by one. 

        Parameters:
            input (list/str): raw values from protein data or antibody name
            transform_type (str): type of transformation. "log" or "arcsinh" 
            transform_scale (int): multiplier applied during transformation
            model (str): type of model to fit. "gaussian" or "nb"
            plot (bool): option to plot each model
            **kwargs: initial arguments for sklearn's GaussianMixture (optional)
            
        Returns:
            gauss_params/nb_params (dict): results from optimization
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
            
        if isinstance(transform_scale, int) == False:
            raise TransformScaleError("'transform_scale' must be an integer value.")

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
            # if input in self._protein_matrix.columns:
            try:
                data_vector = list(self._protein_matrix.loc[:, input].values)
                # Also set ab_name to input, since input is the string of the antibody
                ab_name = input
                individual = True   
            except:
                raise AntibodyLookupError(f"'{input}' not found in protein data.")   
        # Fitting all antibodies at once
        else:
            data_vector = input


        if transform_type is not None:     
            if transform_type == 'log':
                data_vector = _log_transform(d_vect=data_vector,
                                            scale=transform_scale)

            elif transform_type == 'arcsinh':
                data_vector = _arcsinh_transform(d_vect=data_vector,
                                                scale=transform_scale)

            else:
                raise TransformTypeError(("Invalid transformation type. " 
                                          "Please choose 'log' or 'arcsinh'. "
                                          "Default: None."))
            
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
                           **kwargs) -> list:
        """
        Applies fit_antibody to each antibody in the protein count matrix

        Parameters:
            protein_data (pd.DataFrame): count matrix containing antibodies x cells
            transform_type (str): type of transformation. "log" or "arcsinh" 
            transform_scale (int): multiplier applied during transformation
            model (str): type of model to fit. "gaussian" or "nb"
            plot (bool): option to plot each model
            **kwargs: initial arguments for sklearn's GaussianMixture (optional) 

        Returns:
            fit_all_results (list): list of dictionaries, with each dictionary 
            containing optimization results for an antibody
        """

        fit_all_results = []

        for ab in tqdm(self._protein_matrix, total=len(self._protein_matrix.columns)):
            # if plot: # Print antibody name if plotting
                # print("Antibody:", ab)
            fits = self.fit_antibody(input=self._protein_matrix.loc[:, ab], 
                                    ab_name=ab,
                                    transform_type=transform_type,
                                    transform_scale=transform_scale, 
                                    model=model,
                                    plot=plot,
                                    **kwargs)
            self._all_fits_dict[ab] = fits
            fit_all_results.append(fits)

        # Store in class
        self._all_fits = fit_all_results

        return self._all_fits_dict

    def normalize_all_antibodies(self,
                                 p_threshold: float = 0.05,
                                 sig_expr_threshold: float = 0.85,
                                 bg_expr_threshold: float = 0.15,
                                 bg_cell_z_score: int = -10):
        """
        Normalizes all values in a protein matrix

        Parameters:
            p_threshold (float): threshold for p-value rejection when
                performing linear regression to account for cell size and type
            sig_expr_threshold (float): threshold for antibody expression when
                filtering cells that have a high signal expression rate
            bg_expr_threshold (float): threshold for antibody expression when
                filtering cells that have a low signal expression rate
            bg_cell_z_score (int): z-score value for background cells
                when computing a z-score table for all normalized counts
        
        Returns:
            normalized_df (pd.DataFrame): dataframe containing normalized 
                protein values
        """
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
        
        if not bg_cell_z_score < 0:
            raise BackgroundZScoreError("bg_cell_z_score must be less than 0")

        warnings.filterwarnings('ignore')
        # Classify all cells as either background or signal
        classified_cells = _classify_cells_df(all_fits, self._protein_matrix)

        # Filter out cells that have a high signal:background ratio (default: 0.85)
        classified_cells_filt = _filter_classified_df(classified_cells, 
                                                    sig_threshold=sig_expr_threshold,
                                                    bg_threshold=bg_expr_threshold)
        self._classified_filt_df = classified_cells_filt

        # # Filter the same cells from the protein data
        protein_cleaned_filt = _filter_count_df(classified_cells_filt, 
                                                self._protein_matrix)
        
        # Filter from cell labels if dealing with single cell data
        if self._cell_labels is not None:
            cell_labels_filt = _filter_cell_labels(classified_cells_filt, 
                                                        self._cell_labels)
            self._cell_labels_filt_df = cell_labels_filt

        # Calculate z scores for all values
        z_scores = _z_scores_df(all_fits, protein_cleaned_filt)
        self._z_scores_df = z_scores

        # Extract z scores for background cells
        background_z_scores = _bg_z_scores_df(classified_cells_filt, z_scores)

        # If dealing with single cell data with cell_labels,
        # Run linear regression to regress out z scores based on size and cell type
        if self._gene_matrix is not None and self._cell_labels is not None:
            df_by_type = _z_avg_umi_sum_by_type(background_z_scores,
                                                self._gene_matrix,
                                                cell_labels_filt)

            lin_reg_type = _linear_reg_by_type(df_by_type)
            self._linear_reg_df = lin_reg_type

            # Normalize all protein values
            normalized_df = _normalize_antibodies_df(
                                    protein_cleaned_filt_df=protein_cleaned_filt, 
                                    fit_all_results=all_fits,
                                    p_threshold=p_threshold,
                                    background_cell_z_score=bg_cell_z_score,
                                    classified_filt_df=classified_cells_filt, 
                                    cell_labels_filt_df=cell_labels_filt, 
                                    lin_reg_dict=lin_reg_type)
        
        # If dealing with single cell data WITHOUT cell labels:
        # Run linear regression to regress out only size
        elif self._gene_matrix is not None and self._cell_labels is None:
            z_umi = _z_avg_umi_sum(background_z_scores, self._gene_matrix)
            lin_reg = _linear_reg(z_umi)

            # Normalize all protein values
            normalized_df = _normalize_antibodies_df(
                                    protein_cleaned_filt_df=protein_cleaned_filt, 
                                    fit_all_results=all_fits,
                                    p_threshold=p_threshold,
                                    background_cell_z_score=bg_cell_z_score,
                                    classified_filt_df=classified_cells_filt,
                                    lin_reg=lin_reg)

        # Else, normalize values for flow cytometry data
        else:
            # Normalize all values in the protein matrix
            normalized_df = _normalize_antibodies_df(
                                    protein_cleaned_filt_df=protein_cleaned_filt, 
                                    fit_all_results=all_fits,
                                    p_threshold=p_threshold,
                                    background_cell_z_score=bg_cell_z_score,
                                    classified_filt_df=classified_cells_filt)
        
        self._normalized_counts_df = normalized_df

        return normalized_df