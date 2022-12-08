from setuptools import setup, find_packages


if __name__ == '__main__':
    # get text for setup
    # print("\n\n\n starts reading requirements \n\n\n")
    requirements = ["networkx", "torch", "matplotlib", "numpy", "scikit-learn",
                    "tqdm", "graph-measures==0.1.55"]
    # with open("requirements.txt", "r", encoding="utf-8") as f:
    #     requirements = [l.strip() for l in f.readlines()]
    #     print(requirements)
    # print("\n\n\n end reading requirements \n\n\n")

    with open("README.md", "r") as r:
        readme = r.read()

    setup(
        name="EdgeClassifier",
        version="0.0.11",
        license="MIT",
        maintainer="Ziv Naim",
        author="Ziv Naim",
        maintainer_email="zivnaim3@gmail.com",
        url="https://github.com/louzounlab/Edge-Prediction",
        description="A python package for classify edges of graph based "
                    "on topological features and neural networks.",
        long_description=readme,
        long_description_content_type="text/markdown",
        keywords=["gpu", "graph", "edges", "edge", "classification", "neural", "networks"],
        description_file="README.md",
        license_files="LICENSE",
        install_requires=requirements,
        packages=find_packages(),
        python_requires=">=3.6.8",
        package_data={'': ['*.pkl']},
        include_package_data=True,
        has_ext_modules=lambda: True,
        package_dir={"": "."},
        classifiers=[
            'Programming Language :: Python'
        ],
        easy_install="ok_zip"
    )
