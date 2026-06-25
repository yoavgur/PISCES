# Feature sets

A feature set is JSON:
```json
{"name": "...", "description": "...",
 "features": [{"layer": 12, "feature_id": 3456, "sign": -1, "why": "top tokens look like agreement"}]}
```
- `sign = -1` => suppress the feature (maps to PISCES `Feature(neg=True)`).
- `sign = 1`  => the opposite direction.
- Empty `"features": []` is a valid placeholder.

The example files are placeholders. Fill them from the feature-search notebooks (`03`).
