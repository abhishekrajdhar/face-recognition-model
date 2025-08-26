Attendance Recognition API

This project provides a Machine Learning–powered API for an attendance recognition system. Users can capture their image through a mobile or web app, and the backend API verifies the face against stored records to automatically mark attendance.

The API is built with FastAPI for high performance, and the ML model (trained with scikit-learn) handles recognition. It is designed to be lightweight, scalable, and easy to integrate with existing applications such as mobile apps, web dashboards, or classroom management systems.

The service is container-ready and deployable on platforms like Render, where we specify the Python runtime and dependencies for smooth deployment. The API exposes endpoints for user registration, face recognition, and attendance marking.

To run locally, install the dependencies from requirements.txt and start the API server with uvicorn. Once deployed, the API can be consumed by client apps via REST endpoints to streamline attendance management seamlessly.
