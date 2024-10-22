import importlib

# List of libraries to check
libraries = {
    'torch': 'torch',
    'torch.utils.data': 'torch',
    'torch.optim': 'torch',
    'os': 'os',
    'opnn_transformer': 'opnn_transformer',
    'dataset_prep': 'dataset_prep',
    'utils': 'utils',
    'argparse': 'argparse',
    'torch.nn': 'torch',
    'torch.nn.functional': 'torch',
    'json': 'json',
    'datetime': 'datetime',
    'numpy': 'numpy',
    'matplotlib.pyplot': 'matplotlib',
    'torchvision.transforms': 'torchvision'
}

# Check each library
for lib, module in libraries.items():
    try:
        imported_module = importlib.import_module(module)
        if module == 'torch':
            version = imported_module.__version__
        elif module == 'matplotlib':
            version = imported_module.__version__
        elif module == 'numpy':
            version = imported_module.__version__
        elif module == 'torchvision':
            version = imported_module.__version__
        else:
            version = 'version not applicable'
        
        print(f"{lib} - Installed, version: {version}")
    except ImportError:
        print(f"{lib} - NOT installed")
