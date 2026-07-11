from setuptools import setup, find_packages

setup(
    name="pdf2md",
    version="0.1.0",
    author="Maciej Dudek, Matteo Mattiuzzi",
    author_email="Dudek.Maciej@eea.europa.eu, matteo@mattiuzzi.com",
    description="Convert a PDF document into Markdown (.md) or Quarto (.qmd) detecting figures, transcribing tables and writing structured content using LLM APIs.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/eea/pdf2md",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
    python_requires=">=3.10",
    install_requires=[
        "PyMuPDF>=1.26",
        "requests>=2.32",
        "rich>=13.7",
    ],
    entry_points={
        "console_scripts": [
            "pdf2md=pdf2md.app_cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: European Union Public Licence 1.2 (EUPL 1.2)",
        "Operating System :: OS Independent",
    ],
)
