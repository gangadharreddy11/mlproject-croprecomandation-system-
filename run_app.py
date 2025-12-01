# run_app.py
from waitress import serve
from app import app  # your Flask app object

serve(app, host='0.0.0.0', port=8000)
