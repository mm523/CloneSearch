import argparse
from importlib.metadata import version
__version__ = version("clonesearch")

def main():
    parser = argparse.ArgumentParser(
        description="CloneSearch: identifying responding T cell clones" \
                    " from longitudinal TCR sequencing"
    )
    parser.add_argument("--version", action="version", version=f"clonesearch {__version__}")

    subparsers = parser.add_subparsers(
        dest="command",
        title="commands",
        description="Available commands",
    )

    load_parser = subparsers.add_parser(
        "clonesearch-generate_counts_table",
        help="Load TCR sequencing data by timepoint and save a wide-format table which is CloneSearch-friendly. This step is automatically performed by clonesearch-find_outliers.",
    )

    run_parser = subparsers.add_parser(
        "clonesearch-find_outliers",
        help="Use CloneSearch to find the outliers in the timeseries",
    )

    clustering_parser = subparsers.add_parser(
            "clonesearch-clustering",
            help="Cluster TCR trajectories based on their shapes",
        )
    
    args = parser.parse_args()

if __name__ == "__main__":
    main()