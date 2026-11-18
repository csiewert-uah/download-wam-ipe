import os
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

BASE_URL = "https://noaa-nws-wam-ipe-pds.s3.amazonaws.com"
S3_NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}

# Common top-level version prefixes used by NOAA for WAM-IPE
VERSION_PREFIXES = ["v1.2/", "v1.1/", "v1.0/", ""]


def query_s3_files(search_folder: str):
    """
    Search S3 across version prefixes (v1.2/, v1.1/, root) to find files.
    """
    for v_prefix in VERSION_PREFIXES:
        full_prefix = f"{v_prefix}{search_folder}"
        files = fetch_prefix_list(full_prefix)
        if files:
            return files, full_prefix
    return [], None


def fetch_prefix_list(prefix: str):
    file_list = []
    continuation_token = None

    while True:
        params = {"list-type": "2", "prefix": prefix}
        if continuation_token:
            params["continuation-token"] = continuation_token

        url = f"{BASE_URL}/?{urllib.parse.urlencode(params)}"
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )

        try:
            with urllib.request.urlopen(req) as resp:
                xml_data = resp.read()
        except urllib.error.URLError:
            break

        root = ET.fromstring(xml_data)

        for content in root.findall("s3:Contents", S3_NS):
            key = content.findtext("s3:Key", "", S3_NS)
            size = int(content.findtext("s3:Size", "0", S3_NS))
            if key and not key.endswith("/"):
                file_list.append((key, size))

        is_truncated = (
            root.findtext("s3:IsTruncated", "false", S3_NS).lower() == "true"
        )
        if is_truncated:
            continuation_token = root.findtext(
                "s3:NextContinuationToken", None, S3_NS
            )
            if not continuation_token:
                break
        else:
            break

    return file_list


def download_file(key: str, dest_dir: str):
    local_path = os.path.join(dest_dir, key)
    file_url = f"{BASE_URL}/{key}"

    if os.path.exists(local_path):
        print(f"  [Skipped] {key}")
        return

    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    req = urllib.request.Request(
        file_url, headers={"User-Agent": "Mozilla/5.0"}
    )

    try:
        with urllib.request.urlopen(req) as resp, open(
            local_path, "wb"
        ) as out_file:
            while True:
                chunk = resp.read(1024 * 1024)  # 1 MB chunks
                if not chunk:
                    break
                out_file.write(chunk)
        print(f"  [Downloaded] {key}")
    except Exception as e:
        print(f"  [!] Failed to download {key}: {e}")


def main():
    print("=== WAM-IPE AWS NODD Downloader ===")

    # 1. Date Prompts
    start_str = input("Start Date (YYYY-MM-DD): ").strip()
    end_str = input("End Date   (YYYY-MM-DD): ").strip()

    # 2. Cycle Selection Prompt
    cycle_input = (
        input(
            "Forecast cycle (00, 06, 12, 18) [Press Enter for ALL]: "
        )
        .strip()
        .zfill(2)
    )

    # Validate or format cycle input
    valid_cycles = ["00", "06", "12", "18"]
    selected_cycle = cycle_input if cycle_input in valid_cycles else None

    # 3. Save Directory Prompt
    out_dir = (
        input("Save Folder [default: ./wam_ipe_data]: ").strip()
        or "./wam_ipe_data"
    )

    try:
        start_date = datetime.strptime(start_str, "%Y-%m-%d").date()
        end_date = datetime.strptime(end_str, "%Y-%m-%d").date()
    except ValueError:
        print("Error: Please use date format YYYY-MM-DD.")
        return

    current_date = start_date
    while current_date <= end_date:
        date_folder = f"wfs.{current_date.strftime('%Y%m%d')}"

        # Build search path depending on cycle selection
        if selected_cycle:
            folder_pattern = f"{date_folder}/{selected_cycle}/"
        else:
            folder_pattern = f"{date_folder}/"

        print(f"\n[+] Searching for date: {current_date}...")

        files, matched_prefix = query_s3_files(folder_pattern)

        if not files:
            print(f"  [-] No forecast data found for {current_date}")
        else:
            print(
                f"  [+] Found {len(files)} files under '{matched_prefix}'. Downloading..."
            )
            for key, size in files:
                download_file(key, out_dir)

        current_date += timedelta(days=1)

    print("\nDownload complete.")


if __name__ == "__main__":
    main()
