import csv
import utils

def process_2024_house():
    input_file = "2024_house.txt"
    output_file = "house_2024.csv"
    merged_file = "house_margins.csv"

    # Step 1: Read and sort the data by 'abbr'
    with open(input_file, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines if line.strip()]
    header_line = lines[0]
    lines = lines[1:]  # Skip header line
    lines.sort(key=lambda x: x.split()[0])

    # Step 2: Process the data
    processed_data = []
    for line in lines:
        parts = line.split()
        abbr = parts[0]
        R_votes = int(parts[1].replace(",", ""))
        D_votes = int(parts[2].replace(",", ""))
        processed_data.append([abbr, D_votes, R_votes])

    # Step 3: Calculate house_margin, national_margin, and relative_margin
    total_D = sum(row[1] for row in processed_data)
    total_R = sum(row[2] for row in processed_data)
    national_margin = (total_D - total_R) / (total_D + total_R) if (total_D + total_R) > 0 else 0.0

    output_rows = []
    for row in processed_data:
        abbr, D_votes, R_votes = row
        house_margin = (D_votes - R_votes) / (D_votes + R_votes) if (D_votes + R_votes) > 0 else 0.0
        relative_margin = house_margin - national_margin
        output_rows.append([
            2024, abbr, D_votes, R_votes, house_margin, utils.lean_str(house_margin),
            relative_margin, utils.lean_str(relative_margin),
            national_margin, utils.lean_str(national_margin)
        ])

    # Step 4: Save to house_2024.csv
    with open(output_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "year", "abbr", "D_votes", "R_votes", "house_margin", "house_margin_str",
            "relative_margin", "relative_margin_str", "national_margin", "national_margin_str"
        ])
        writer.writerows(output_rows)

    # Step 5: Merge with house_margins.csv
    with open(merged_file, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        existing_data = list(reader)

    merged_data = existing_data + output_rows
    merged_data.sort(key=lambda x: (int(x[0]), x[1]))  # Sort by year, then abbr

    with open(merged_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(merged_data)

    print(f"Processing complete. Data saved to {output_file} and merged into {merged_file}.")

if __name__ == "__main__":
    process_2024_house()
