import os

def get_image_files(index_lst: list) -> list:
    """
    This function takes a list of indices and returns a list of all image files found in the corresponding directories.
    
    Parameters:
    index_lst (list): A list of indices. Each index is used to construct a directory path to search for image files.
    
    Returns:
    list: A list of all image files found in the directories specified by the indices.
    """
    image_files = []

    for i in index_lst:
        image_dir = f"D:\BigData\images_00{i}\images"
        
        image_files_for_one_folder = os.listdir(image_dir)
        
        image_files += image_files_for_one_folder
        
    return image_files