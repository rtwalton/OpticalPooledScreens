import os
import ops
from ops.imports import *

# Set screen directories
parse_function_home = "/lab/barcheese01/screens"
parse_function_dataset = "baker"

# Set home directory as a combination of parse_function_home and parse_function_dataset
home = os.path.join(parse_function_home, parse_function_dataset)

# Change to the home directory
os.chdir(home)

# Read in data
df_cp_phenotype = pd.read_hdf('ph_2/cp_phenotype.hdf')
df_merged_deduped = pd.read_hdf('merge_3/merged_deduped.hdf')

# Print/save
print(df_cp_phenotype.head())
print(df_cp_phenotype.columns)
df_cp_phenotype.head(100).to_csv('df_cp_phenotype_test.csv', index=False)

print(df_merged_deduped.head())
print(df_merged_deduped.columns)
df_merged_deduped.head(100).to_csv('df_merged_deduped_test.csv', index=False)

# Merge df_merged_deduped and df_cp_phenotype
single_cell_phenotype = df_merged_deduped.merge(df_cp_phenotype.rename(
    columns={'label':'cell_0'}), how='left',on=['well','tile','cell_0'])

# Print/save
print(single_cell_phenotype.head())
print(single_cell_phenotype.columns)
single_cell_phenotype.head(100).to_csv('single_cell_phenotype_test.csv', index=False)

single_cell_phenotype.to_hdf('single_cell_phenotype.hdf','x',mode='w')









### START HERE:
# # Read in data
# df_cp_phenotype_subset = pd.read_hdf('single_cell_phenotype.hdf')

# # Find the starting index of the columns you want to consider
# start_column_index = df_cp_phenotype_subset.columns.get_loc("nucleus_dapi_int")

# # Select columns from 'nucleus_dapi_int' onward
# selected_columns = list(df_cp_phenotype_subset.columns[start_column_index:])

# # Subset the DataFrame to include only the columns of interest
# selected_columns = ['gene_symbol_0'] + selected_columns

# # Subset the DataFrame to include only the columns of interest
# subset_df = df_cp_phenotype_subset[selected_columns]

# # Set 'gene_symbol_0' as the index
# subset_df.set_index('gene_symbol_0', inplace=True)

# # Calculate the number of rows corresponding to each unique index
# row_counts = pd.DataFrame(subset_df.index.value_counts())

# # Rename the only column to "cell_number"
# row_counts.columns = ['cell_number']

# # Save it as a CSV
# row_counts.to_csv('row_counts.csv', index=True)

# # Calculate the median for each unique 'gene_symbol_0'
# median_df = subset_df.median(level='gene_symbol_0')

# # Print the result
# median_df.to_csv('simple_medians.csv', index=True)

# # Filter rows where "gene_symbol_0" begins with "sg_nt"
# filtered_rows = df_cp_phenotype_subset[df_cp_phenotype_subset["gene_symbol_0"].str.startswith("sg_nt")]

# # Group by "well" on the filtered rows
# grouped_filtered_rows = filtered_rows.groupby("well")

# # Calculate median and median standard deviation for selected columns
# result_df = grouped_filtered_rows[selected_columns].agg(["median", "std"])

# # Print the result
# print("Median and Median Standard Deviation for selected columns (filtered rows):")
# print(result_df)

# result_df.to_csv("cp_phenotype_nt_median_std.csv", index=True)

# # Group by "well" on the filtered rows
# df_cp_phenotype_subset_rows = df_cp_phenotype_subset.groupby("well")

# print(df_cp_phenotype_subset_rows)

# # Normalize each column based on the median and std from result_df
# for well, well_data in df_cp_phenotype_subset_rows:
#     for column in selected_columns:
#         # Extract median and std values for the current column and well
#         median_val = result_df.loc[well, (column, "median")]
#         std_val = result_df.loc[well, (column, "std")]
#         print(well,column)

#         # Normalize the current column in df_cp_phenotype_subset
#         df_cp_phenotype_subset.loc[well_data.index, column] = (well_data[column] - median_val) / std_val

# # Save the normalized DataFrame to a new CSV file
# df_cp_phenotype_subset.head(100).to_csv('df_cp_phenotype_subset_normalized.csv', index=False)