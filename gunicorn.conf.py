"""Gunicorn configuration for Zeabur deployment."""

bind = "0.0.0.0:8080"
workers = 2
timeout = 300  # Claude API 回應較慢，需要較長 timeout
