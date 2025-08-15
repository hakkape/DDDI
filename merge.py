import polars as pl
import glob
import os
import argparse

def merge_csv_files(output_path: str, *patterns: str):
    """Merge CSV files matching the given glob patterns into output_path.

    parameters:
    - output_path: absolute or relative path to the output CSV file
    - patterns: variable number of glob pattern strings. For backward compatibility,
      a single iterable of strings (list/tuple/set) is also accepted as the only argument.
    """
    # Backward compatibility: allow a single iterable passed as the only vararg
    if len(patterns) == 1 and isinstance(patterns[0], (list, tuple, set)):
        pattern_list = list(patterns[0])
    else:
        pattern_list = list(patterns)

    csv_files = [os.path.abspath(f) for p in pattern_list for f in glob.glob(p)]
    frames = []

    output_path = os.path.abspath(output_path)

    for f in csv_files:
        if f == output_path:
            print(f"Skipping output file: {os.path.basename(output_path)}")
            continue

        df = pl.read_csv(f, infer_schema=False)
        frames.append(df)
        print(f"Loaded: {os.path.basename(f)}")

    if frames:
        try:
            result = pl.concat(frames)
            result.write_csv(output_path)
            print(f"All files merged into {os.path.basename(output_path)}")
        except Exception as e:
            print(f"Error during concatenation: {e}")
    else:
        print("No CSV files found matching the pattern.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge all CSV files for a given path")
    parser.add_argument('pattern', nargs='+', default='*.csv', help="File pattern to match CSV files (default: '*.csv')")
    parser.add_argument('-o', '--output', default='join.csv', help="Output filename (default: 'join.csv')")
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)

    if os.path.exists(output_path):
        confirm = input(f"Output file {args.output} already exists. Overwrite? (y/N): ").strip().lower()
        if confirm != 'y':
            print("Aborted. Output file not overwritten.")
            exit()

    merge_csv_files(output_path, *args.pattern)
