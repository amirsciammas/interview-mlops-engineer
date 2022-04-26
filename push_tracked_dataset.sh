# Steps to track and push to a S3 bucket a dataset with DVC

# Install DVC if needed
# pip install "dvc[s3]"

# Initialize DVC repository
dvc init
git commit -m "Initialize DVC repository"
echo "Initialized DVC repository"

# Configure S3 remote storage, let's suppose it is s3://adidastestavillalba/data/
dvc remote add -d storage s3://adidastestavillalba/data/
git add .dvc/config
git commit -m "Created remote storage"
echo "Created a S3 remote storage for the dataset"

# Track the dataset
dvc add dataset.csv
git add dataset.csv.dvc .gitignore
git commit -m "[Dataset] Track dataset with DVC"
echo "Tracking dataset"

# Now that the dataset is tracked and versioned, push the current version to S3
dvc push
echo "Dataset has been pushed to the configured S3 bucket"

# Changes in the dataset can be pulled by doing:
# dvc pull