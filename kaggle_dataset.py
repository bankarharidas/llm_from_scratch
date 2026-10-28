import kagglehub

# Download latest version
path = kagglehub.dataset_download("shadow4ever/top-112-hindi-swear-words")

print("Path to dataset files:", path)
