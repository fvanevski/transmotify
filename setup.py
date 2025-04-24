import setuptools

# Read the contents of your README file
# (Optional, but good for PyPI)
try:
    with open("README.md", "r", encoding="utf-8") as fh:
        long_description = fh.read()
except FileNotFoundError:
    long_description = "Your project description"  # Fallback if README.md is not found

setuptools.setup(
    # --- Essential Arguments ---
    name="transmotify",  # Replace with the actual name of your package (e.g., 'transmotify')
    version="0.1.0",  # Start with a version number
    packages=setuptools.find_packages(),  # Automatically find all packages (directories with __init__.py)
    # --- Dependencies ---
    # List your project's dependencies required for installation
    install_requires=[
        # Examples based on your previous comments.
        # List ALL libraries your project needs to run here.
        # 'pytest', # pytest is usually a dev/test dependency, not required for runtime
        # 'rich',   # rich might or might not be required for runtime
        # Add actual dependencies here, e.g.:
        "requests>=2.28.1",  # Example dependency
        # Any libraries your 'io' module or other parts of the project import
    ],
    # --- Metadata (Highly Recommended) ---
    author="Your Name",  # Replace with your name
    author_email="your.email@example.com",  # Replace with your email
    description="A short description of your project",  # Short description
    long_description=long_description,  # Use the README content
    long_description_content_type="text/markdown",  # Specify that long_description is Markdown
    url="https://github.com/yourusername/yourproject",  # Optional: Link to your repo or project page
    classifiers=[
        # Choose classifiers from https://pypi.org/classifiers/
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",  # Example license
        "Operating System :: OS Independent",
    ],
    # --- Specify Python Version Compatibility ---
    python_requires=">=3.8",  # Set the minimum Python version required
    # You might set this higher, e.g., '>=3.11' based on your Conda env
)
