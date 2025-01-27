import os

def fun_walk_through_dir(directory_path):
    """
    Function to walk through a directory and count the files within each subfolder.
    Only includes subfolders that contain files.
    The table headers are dynamically generated based on subfolders and categories.
    """
    category_counts = {}

    # Walk through all directories and subdirectories within the main directory
    for root, dirs, files in os.walk(directory_path):
        # Extract the name of the top-level folder (category)
        category = root.split(os.path.sep)[-2]  # The category is the second-to-last folder
        subfolder = root.split(os.path.sep)[-1]  # The subfolder is the last folder

        # If there are files in the subfolder, process the count
        if files:
            # Ensure the subfolder is in the dictionary
            if subfolder not in category_counts:
                category_counts[subfolder] = {}

            # If the category is not in the dictionary for that subfolder, add it
            if category not in category_counts[subfolder]:
                category_counts[subfolder][category] = 0

            # Count the files in the subfolder
            file_count = len(files)

            # Store the file count in the dictionary
            category_counts[subfolder][category] = file_count

    # Display the full summary of the number of files in each subfolder
    print("Summary of files by subfolder:")
    print("-" * 100)

    # Generate the headers for the columns (categories), only those that have files
    all_categories = sorted({category for subfolder in category_counts.values() for category in subfolder.keys()})

    # Print the table headers
    print(f"{'Subfolder':<30} " + "  ".join([f"{category:<15}" for category in all_categories]))
    print("-" * 100)

    # Print the rows of the table with the file counts
    for subfolder, categories in category_counts.items():
        row = f"{subfolder:<30}"
        for category in all_categories:
            # If the category does not exist for a subfolder, show 0
            row += f"  {categories.get(category, 0):<15}"
        print(row)

    return category_counts

# # Call the function with the path where the data is extracted EXAMPLE
# dataset_path = "/content/dataset/10_food_classes_10_percent"
# category_counts = fun_walk_through_dir_v6(dataset_path)
