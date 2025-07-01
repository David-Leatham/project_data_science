Server starten mit: ```uvicorn main:app --reload```
Test with: ```Invoke-WebRequest -Uri "http://127.0.0.1:8000/fraud-prediction" -Method POST -ContentType "application/json" -InFile "fraud_request_body.json"```
           ```Invoke-WebRequest -Uri "http://127.0.0.1:8000/fraud-prediction" -Method POST -ContentType "application/json" -InFile "request_body.json"```