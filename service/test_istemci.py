import sys
import requests

yol = sys.argv[1]
with open(yol, "rb") as f:
    r = requests.post("http://127.0.0.1:8000/predict",
                      files={"file": f},
                      params={"include_heatmap": False})
print(r.status_code, r.json())