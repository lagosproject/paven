import os

def count_images(folder):
    image_count = 0
    for file in os.listdir(folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            image_count += 1
    return image_count

def count_images_in_subfolders(folders):
    total_counter = 0
    for folder in folders:
        dataset_counter = 0
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        for subfolder in subfolders:
            image_count = count_images(subfolder)
            if image_count > 32:
                #print(f"Folder: {subfolder}, Image Count: {image_count}")
                dataset_counter += image_count
        print(f"Total images in {folder}: {dataset_counter}")
        total_counter += dataset_counter
    print(f"Total images in all folders: {total_counter}")
    return total_counter

def count_images_in_subfolders_strucutre(folders):
    total_counter = 0
    for folder in folders:
        dataset_counter = 0
        subfolders = [f.path for f in os.scandir(folder) if f.is_dir()]
        for subfolder in subfolders:
            image_count = count_images(subfolder+"/frames")
            if image_count > 32:
                #print(f"Folder: {subfolder}, Image Count: {image_count}")
                dataset_counter += image_count
        print(f"Total images in {folder}: {dataset_counter}")
        total_counter += dataset_counter
    print(f"Total images in all folders: {total_counter}")
    return total_counter

# Example usage
folders = [
    "/media/beegfs/home/v582/PROJECT/DHF1K/videoframes",
    "/media/beegfs/home/v582/PROJECT/EyeFixationMaps/SlicedVideos",
    #"/media/beegfs/home/v582/PROJECT/HVECEyeTracking/frames",

    "/media/beegfs/home/v582/PROJECT/AVS1K/trainSet/Frame",
    "/media/beegfs/home/v582/PROJECT/AVS1K/validSet/Frame",

    "/media/beegfs/home/v582/PROJECT/MVVA/mvva_database_v1/frames",

    "/media/beegfs/home/v582/PROJECT/AViNet/video_frames/AVAD", 
    "/media/beegfs/home/v582/PROJECT/AViNet/video_frames/Coutrot_db1", 
    "/media/beegfs/home/v582/PROJECT/AViNet/video_frames/Coutrot_db2",
    "/media/beegfs/home/v582/PROJECT/AViNet/video_frames/DIEM",
    "/media/beegfs/home/v582/PROJECT/AViNet/video_frames/ETMD_av",
    "/media/beegfs/home/v582/PROJECT/AViNet/video_frames/SumMe",
    ]
folder_with_subfolder_strucuture = [
    "/media/beegfs/home/v582/PROJECT/LEDOV/LEDOV",
    "/media/beegfs/home/v582/PROJECT/MVS/MVS"
    ]
total1 = count_images_in_subfolders(folders)
total2 = count_images_in_subfolders_strucutre(folder_with_subfolder_strucuture)

print(f"Total images in all folders: {total1+total2}")