#!/bin/bash
# Create main project directories
mkdir -p src/{data/{raw,processed},models/{towers,layers},trainers,utils,notebooks}
touch src/data/__init__.py
touch src/models/__init__.py
touch src/trainers/__init__.py
touch src/utils/__init__.py 