---

**CONTEXT:**
We are Team BrainRot participating in the NASA International Space Apps Challenge 2025 (October 4-5, 2025). We're working on the challenge "A World Away: Hunting for Exoplanets with AI".

**PROJECT OVERVIEW:**
- **Objective:** Create an AI/ML model trained on NASA's open-source exoplanet datasets (Kepler, TESS, K2) to analyze new data and accurately identify exoplanets
- **Deliverable:** Web application with machine learning backend and interactive frontend
- **Tech Stack:** Flask (Python backend) + React (frontend) + ReactBits UI components
- **Timeline:** 48-hour hackathon (October 4-5, 2025)

**CURRENT PROGRESS:**
 nothing has been completed yet only structure is being discussed
**SPECIFIC CHALLENGE:**
while my team is working on backend and collecting data, ai, data base etc i want to start working on frontend first from landing page with hero section being with this eye catching background https://reactbits.dev/backgrounds/galaxy after that we as we scroll down we get the typical content we get on landing page of a website

**TECHNICAL REQUIREMENTS:**
- Must work with NASA exoplanet transit data (light curves)
- Web interface for data upload and prediction display
- Real-time model inference
- Responsive design with space theme
- Deployment ready for demo

**CONSTRAINTS:**
- Hackathon environment (limited time)
- Open-source solutions only
- Must use NASA's publicly available datasets
- Team has experience with Flask + React integration challenges

**EXPECTED OUTPUT:**
Help design frontend feature by feature, step by step, page by page, basic to advance while keeping in mind that ai will be integrated and backend as flask
Using the context above, we need help building [specific React component/feature]. We're using ReactBits library and want to create [specific functionality]. Please provide React code with modern hooks.

Structure
exoplanet-detector/
├── backend/
│   ├── app.py                 # Flask main
│   ├── models/
│   │   ├── exoplanet_detector.py
│   │   └── data_preprocessor.py
│   ├── api/endpoints.py
│   └── data/processed/
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ui/         # ReactBits components
│   │   │   └── charts/     # Visualization
│   │   ├── pages/
│   │   │   ├── Dashboard.jsx
│   │   │   ├── DataUpload.jsx
│   │   │   └── Results.jsx
│   │   └── services/api.js
│   └── package.json
└── deployment/
    ├── Dockerfile
    └── requirements.txt

---

