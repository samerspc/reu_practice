# Food Recognition Web App

Image recognition web application using a Keras model, Flask backend, and React frontend.

## Setup

1.  **Model File:** Place `model_mobilenet_final.h5` in the root directory.
2.  **Prerequisites:** Install Docker Desktop.

## Run Application

From the project root, run:

```bash
docker-compose up --build
```

Access frontend at `http://localhost:3000`.

## Details

-   **Backend:** Flask (Python) on host port `5001`.
-   **Frontend:** React.js served by Nginx on host port `3000`.

---
