from snmfem.datasets import generate_edxs_dataset, generate_toy_dataset

if __name__ == "__main__":
    generate_edxs_dataset(seeds=range(10))
    generate_toy_dataset(seeds=range(10))
