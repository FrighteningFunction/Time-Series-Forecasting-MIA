curl -X POST https://api.runpod.ai/v2/$ENDPOINT_ID/runsync \
          -H "Content-Type: application/json" \
          -H "Authorization: Bearer $RUNPOD_API_KEY" \
          -d '{
  "input_data": {
    "num_runs": 1,
    "epochs": 2,
    "num_shadow": 2,
    "batch_size": 64,
    "output_dir": "artifacts/flash_remote_smoke"
  }
}
'