# 🎬 Movie Recommendation System

## 🚀 Overview

This project is an advanced **Movie Recommendation System** that suggests movies to users based on their preferences and behavior.

The system combines multiple machine learning techniques to provide accurate and personalized recommendations:

- 🔥 Weighted Collaborative Filtering
- 🎯 Content-Based Filtering (TF-IDF)
- 🤝 Hybrid Recommendation System
- 📊 K-Nearest Neighbors (KNN)

An interactive web application is built using **Streamlit**.

---

## 🧠 Key Features

- Personalized movie recommendations
- Handles cold-start problem (new users)
- Hybrid recommendation (combines multiple methods)
- Efficient computation using sparse matrices
- Fast predictions using saved models (pickle)
- User-friendly interface with Streamlit

---

## ⚙️ Tech Stack

**Programming Language:**
- Python

**Libraries Used:**
- Pandas
- NumPy
- Scikit-learn
- SciPy
- Streamlit

---

## 🏗️ Project Structure


movie-recommender/
│
├── app.py # Streamlit application
├── train_model.py # Model training script
├── requirements.txt # Required libraries
├── README.md # Project documentation
├── .gitignore


---

## 🧪 Models Used

### 1. Collaborative Filtering (User-Based)
- Uses cosine similarity between users
- 🔥 Weighted averaging for better accuracy
- Recommends movies liked by similar users

---

### 2. Content-Based Filtering
- Uses movie genres
- Applies TF-IDF vectorization
- Finds similar movies using cosine similarity

---

### 3. Hybrid Recommendation System
- Combines:
  - Collaborative Filtering
  - Content-Based Filtering
- Produces more robust recommendations

---

### 4. KNN Model (Item-Based)
- Uses Nearest Neighbors algorithm
- Finds similar movies based on user ratings

---

## 📊 Evaluation Metrics

The system is evaluated using:

- **Precision@K** → Measures relevance of recommended movies
- **Recall@K** → Measures how many relevant movies are recommended

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository


git clone https://github.com/your-username/movie-recommender.git
cd movie-recommender


---

### Step 2: Install Dependencies


pip install -r requirements.txt


---

### Step 3: Train the Model


python train_model.py


---

### Step 4: Run the Application


python -m streamlit run app.py


---

## 💡 Example Usage

- Enter a **User ID** → Get personalized recommendations  
- Enter a **Movie Name** → Get similar movies  
- Use **Hybrid Mode** → Best combined results  
- Use **KNN Mode** → Item-based recommendations  

---

## 🚀 Future Improvements

- Add movie posters using external APIs
- Deploy application on cloud (Streamlit Cloud / Render)
- Improve recommendations using Deep Learning
- Add search suggestions (autocomplete)

---

## 👨‍💻 Author

Lade Prahlad
GitHub: https://github.com/LadePrahlad

---

## ⭐ Support

If you found this project useful, consider giving it a ⭐ on GitHub!