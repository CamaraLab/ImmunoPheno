import pandas as pd


class STvEA:
    hdbscan_scans = list()

    cite_latent = pd.DataFrame()
    cite_mRNA = pd.DataFrame()
    cite_protein = pd.DataFrame()
    cite_emb = pd.DataFrame()
    cite_cluster = pd.DataFrame()
    cite_cluster_name_dict = dict()

    codex_blanks = pd.DataFrame()
    codex_protein = pd.DataFrame()
    codex_protein_corrected = pd.DataFrame()
    codex_size = pd.DataFrame()
    codex_spatial = pd.DataFrame()
    codex_emb = pd.DataFrame()
    codex_knn = pd.DataFrame()
    codex_cluster = pd.DataFrame()
    codex_cluster_names_transferred = pd.DataFrame()
    codex_cluster_name_dict = dict()
    codex_mask = pd.Series()

    transfer_matrix = pd.DataFrame()
    nn_dist_matrix = pd.DataFrame()