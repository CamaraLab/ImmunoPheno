# Removing ADT_ labels from CITE-seq protein data
def clean_adt(cite_protein_df):
    '''
    Input:
      cite_protein_df: pandas dataframe, cite-seq protein data (adt/protein x sample)
    
    Output:
      cite_protein_df_copy: pandas dataframe, adt tags removed from each adt
    '''

    # Deep copy of original cite_protein dataframe
    cite_protein_df_copy = cite_protein_df.copy(deep=True)
    
    # Temporary move row headers into first column by removing the index
    cite_protein_df_copy.reset_index(inplace=True)

    # Remove ADT prefix and return new dataframe
    cite_protein_df_copy.iloc[:,0] = cite_protein_df_copy.iloc[:,0].apply(lambda x: re.sub("ADT_", "", x))

    # Make the (new) first column the new index now 
    # Transpose matrix
    cite_protein_df_copy = cite_protein_df_copy.T

    # Set column labels equal to values in 2nd row
    cite_protein_df_copy.columns = cite_protein_df_copy.iloc[0]

    # Remove 2nd row
    cite_protein_df_copy.drop(cite_protein_df_copy.index[0], inplace=True)

    # Return transposed copy
    # The first row will be the column index
    return cite_protein_df_copy