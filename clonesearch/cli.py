import argparse
from importlib.metadata import version
__version__ = version("clonesearch")

def main():
    parser = argparse.ArgumentParser(
        description="CloneSearch: identifying responding T cell clones" \
                    " from longitudinal TCR sequencing"
    )
    parser.add_argument("--version", action="version", version=f"clonesearch {__version__}")
    
    args = parser.parse_args()

if __name__ == "__main__":
    main()