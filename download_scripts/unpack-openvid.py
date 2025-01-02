from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import argparse

def unzip_single_file(i, zip_folder, video_folder, error_log_path):
    """
    Unzips a single file "OpenVid_part{i}.zip" from zip_folder to video_folder.
    If that fails, attempts to concatenate partaa/partab first, then unzip again.
    """
    file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")

    if not os.path.exists(file_path):
        print(f"File {file_path} does not exist, skipping/unable to unzip.")
        return

    unzip_command = ["unzip", "-j", file_path, "-d", video_folder]
    try:
        subprocess.run(unzip_command, check=True)
        print(f"Successfully unzipped {file_path} to {video_folder}")
    except subprocess.CalledProcessError as e:
        error_message = f"Error unzipping {file_path}: {e}\n"
        print(error_message)
        with open(error_log_path, "a") as error_log_file:
            error_log_file.write(error_message)

        part_uris = [
            f"OpenVid_part{i}_partaa",
            f"OpenVid_part{i}_partab"
        ]
        for part_uri in part_uris:
            part_file_path = os.path.join(zip_folder, os.path.basename(part_uri))
            if not os.path.exists(part_file_path):
                print(f"file {part_file_path} does not exist, skipping.")

        # Merge parts
        cat_command = (
            "cat " +
            os.path.join(zip_folder, f"OpenVid_part{i}_part*") +
            " > " +
            file_path
        )
        os.system(cat_command)

        try:
            subprocess.run(unzip_command, check=True)
            print(f"Successfully unzipped {file_path} (parts) to {video_folder}")
        except subprocess.CalledProcessError as e2:
            second_error = f"Second unzip attempt failed for {file_path}: {e2}\n"
            print(second_error)
            with open(error_log_path, "a") as error_log_file:
                error_log_file.write(second_error)


def unzip_files(zip_directory, output_directory, max_workers=32):
    """
    Unzips all OpenVid_part{i}.zip files in parallel using max_workers threads.
    Also downloads CSV data afterward.
    """
    zip_folder = zip_directory
    video_folder = os.path.join(output_directory, "video")
    os.makedirs(video_folder, exist_ok=True)

    error_log_path = os.path.join(zip_folder, "download_log.txt")

    futures = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i in range(186):
            file_path = os.path.join(zip_folder, f"OpenVid_part{i}.zip")
            if not os.path.exists(file_path):
                print(f"File {file_path} does not exist, skipping.")
                continue

            futures.append(
                executor.submit(
                    unzip_single_file,
                    i,
                    zip_folder,
                    video_folder,
                    error_log_path
                )
            )

        for future in as_completed(futures):
            future.result()

    data_folder = os.path.join(output_directory, "data", "train")
    os.makedirs(data_folder, exist_ok=True)
    data_urls = [
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVid-1M.csv",
        "https://huggingface.co/datasets/nkp37/OpenVid-1M/resolve/main/data/train/OpenVidHD.csv"
    ]
    for data_url in data_urls:
        data_path = os.path.join(data_folder, os.path.basename(data_url))
        command = ["wget", "-O", data_path, data_url]
        subprocess.run(command, check=True)
        print(f"Downloaded {data_url} to {data_path}")


def main():
    unzip_files(
        zip_directory="/datasets/nkp37/OpenVid-1M",
        output_directory="/datasets/OpenVid-1M",
        max_workers=32
    )

if __name__ == "__main__":
    main()
