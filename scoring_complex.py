import os
import numpy as np
import matplotlib.pyplot as plt
import pathlib
def cosine_similarity(normals_1, normals_2, epsilon=1e-8):
    dot_product = np.sum(normals_1 * normals_2, axis=-1)
    norm_1 = np.linalg.norm(normals_1, axis=-1)
    norm_2 = np.linalg.norm(normals_2, axis=-1)
    
    # Adding epsilon to avoid division by zero
    similarity = dot_product / (norm_1 * norm_2 + epsilon)
    return similarity

# Path to the folder containing subdirectories
results_folder = pathlib.Path("/home/screwdriver/Experiments/DSINE/data/results_")
# Output file for cosine similarity
output_file = '/home/screwdriver/Experiments/DSINE/data/results/cosine_similarity.txt'
        
# Loop through all directories in the folder
for subdir in results_folder.iterdir():
    if subdir.is_dir():  # Check if it's a directory
        print(f"Processing folder: {subdir}")
        
        # Directories
        folder_1 = f'{subdir}/DINE_npy_OUTPUT'
        folder_2 = f'{subdir}/mesh_npy_OUTPUT'
        mask_path = f'{subdir}/mask'
        # Get the list of files in both folders
        files_1 = sorted(os.listdir(folder_1))
        files_2 = sorted(os.listdir(folder_2))
        mask_files = sorted(os.listdir(mask_path))

        # Ensure all directories have the same number of files
        assert len(files_1) == len(files_2) == len(mask_files), "The number of files in both folders and mask must be the same."
        # Open the output file for writing cosine similarities
        with open(output_file, 'a') as f:
            f.write(f"{files_1}\tmean_cos_sim\tmedian_cos_sim\n")  # Header for the output file
            
            # Iterate over the files in both folders
            for file_1, file_2, mask_file in zip(files_1, files_2, mask_files):
                # Ensure all files have the same name
                assert file_1 == file_2 == mask_file, f"File names do not match: {file_1} vs {file_2} vs {mask_file}"

                # Load the numpy arrays from each folder
                normals_1 = np.load(os.path.join(folder_1, file_1))
                normals_2 = np.load(os.path.join(folder_2, file_2))
                mask = np.load(os.path.join(mask_path, mask_file))

                # Reshape if needed: Convert shape (1, 3, 800, 800) to (800, 800, 3)
                if normals_2.shape == (1, 3, 800, 800):
                    normals_2 = np.transpose(normals_2[0], (1, 2, 0))  # (1, 3, 800, 800) -> (800, 800, 3)
                
                if normals_1.shape == (1, 3, 800, 800):
                    normals_1 = np.transpose(normals_1[0], (1, 2, 0))  # (1, 3, 800, 800) -> (800, 800, 3)

                # The mask should be applied to the similarity calculation
                # Mask is assumed to be (H, W, 3), so we take only one channel (grayscale mask)
                mask_single_channel = mask[..., 0]  # Use the first channel if the mask is 3D
                
                # Compute cosine similarity
                cos_sim = cosine_similarity(normals_1, normals_2)

                # Apply the mask to cosine similarity (where mask is True, the cosine similarity will be ignored)
                cos_sim_masked = np.ma.masked_array(cos_sim, mask=mask_single_channel)

                # Compute mean and median cosine similarity only in the valid (non-masked) region
                mean_cos_sim = np.ma.mean(cos_sim_masked)
                median_cos_sim = np.ma.median(cos_sim_masked)

                # Visualize the cosine similarity using a heatmap
                plt.figure(figsize=(10, 8))
                plt.imshow(cos_sim_masked, cmap='coolwarm', vmin=0.8, vmax=1)
                plt.colorbar(label="Cosine Similarity")
                plt.title(f"Cosine Similarity Heatmap - {file_1}")

                # Save the heatmap as an image file
                output_directory = "/home/screwdriver/Experiments/DSINE/data/heapmaps"
                output_file_path = os.path.join(output_directory, f"{subdir}{file_1}_heatmap.png")
                plt.savefig(output_file_path, dpi=300, bbox_inches='tight')  # Saves the image as PNG with 300 DPI
                plt.close()  # Close the figure to avoid showing during iteration

                # Write the result to the output file
                f.write(f"{file_1}\t{mean_cos_sim:.6f}\t{median_cos_sim:.6f}\n")

print(f"Cosine similarities computed and written to {output_file}")