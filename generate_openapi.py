import json
from main import app

openapi_schema = app.openapi()

with open("openapi.json", "w") as f:
    json.dump(openapi_schema, f, indent=2)

print("OpenAPI spec saved to openapi.json")
