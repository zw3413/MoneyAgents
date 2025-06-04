#!/bin/bash

if [ "${DEBUGGER:-0}" = "1" ]; then
    echo "🔍 Starting with debugger enabled on port 5678"
    python -m debugpy --listen 0.0.0.0:5678 --wait-for-client -m uvicorn trading_asst.api.main:app --host 0.0.0.0 --port 8080 --reload
else
    echo "🚀 Starting normally"
    uvicorn trading_asst.api.main:app --host 0.0.0.0 --port 8080 --reload
fi 