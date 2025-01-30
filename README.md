# Voiture AI Backend

## Prerequisites

- Python 3.x installed
- Pip 25.0 installed
- Git installed

## Installation Guide

### 1. Clone the Repository

```sh
git clone https://github.com/ChiuDave/Voiture_AI_Backend.git
cd Voiture_AI_Backend
```

### 2. Create a Virtual Environment

#### Windows:

```sh
python -m venv venv
```

#### Linux/Mac:

```sh
python3 -m venv venv
```

### 3. Activate the Virtual Environment

#### Windows:

```sh
venv\Scripts\activate
```

#### Linux/Mac:

```sh
source venv/bin/activate
```

### 4. Install Dependencies

```sh
pip install -r requirements.txt
```

## Running the Application

For the first-time run, execute the following commands:

```sh
python generate_embeddings.py
python main.py
```

For subsequent runs, you can start the application with:

```sh
python main.py
```

## Deactivating the Virtual Environment

To deactivate the virtual environment, run:

```sh
deactivate
```

