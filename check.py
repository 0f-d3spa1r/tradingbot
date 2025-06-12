import os
file_path = "xgboost_model.json"
if os.path.exists(file_path):
    print("File exists")
else:
    print("File not found")