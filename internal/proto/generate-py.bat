@echo off
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. llmserver.proto
python fix_imports.py 