# 🎬 Movie Recommendation System

## 📌 Project Overview
This project builds a **Movie Recommendation System** using collaborative filtering techniques. It provides two types of recommendations:

- **User-based Collaborative Filtering (SVD):** Recommends movies to users based on their past ratings.
- **Item-based Collaborative Filtering (KNN):** Finds similar movies based on rating patterns.

It leverages **scikit-surprise** for model training and **ipywidgets** for an interactive UI.

## 📊 Dataset
The system uses the **MovieLens 100K dataset**, which contains information about user ratings and movie details:

- `userId` (Unique user identifier)
- `movieId` (Unique movie identifier)
- `rating` (User rating of the movie: 1-5 scale)
- `title` (Movie title)

## ⚙️ Machine Learning Approach

### 1️⃣ Data Preprocessing
- Merged ratings with movie titles.
- Converted data into the Surprise library format.

### 2️⃣ **Model Training**
Trained two models for recommendations:
- **Singular Value Decomposition (SVD)** for user-based recommendations.
- **K-Nearest Neighbors (KNN)** for item-based recommendations.

### 3️⃣ **Evaluation Metrics**
- **Root Mean Square Error (RMSE)**

## 🛠 Installation & Usage

### 1️⃣ Install Dependencies
```bash
pip install numpy pandas joblib scikit-surprise ipywidgets
```

### 2️⃣ Run the Model
```python
python MovieRecommendation.py
```

### 3️⃣ Make Predictions

#### **Get Movie Recommendations for a User**
```python
import joblib
from movie_recommender import get_movie_recommendations

# Load trained model
best_svd = joblib.load("movie_recommender.pkl")
movies = pd.read_csv("u.item", sep="|", encoding="ISO-8859-1", names=["movieId", "title"], usecols=[0,1])

# Get recommendations
recommended_movies = get_movie_recommendations(user_id=1, model=best_svd, movies_df=movies, n=5)
print("Recommended Movies:", recommended_movies)
```

#### **Find Similar Movies Using KNN**
```python
from movie_recommender import get_similar_movies

# Load trained KNN model
knn = joblib.load("knn_movie_recommender.pkl")

# Get similar movies
similar_movies = get_similar_movies(movie_id=10, knn_model=knn, movies_df=movies, n=5)
print("Similar Movies:", similar_movies)
```

## 🚀 Future Improvements
- Implement **deep learning** for recommendation (Neural Networks).
- Deploy as a **Flask API** or **Streamlit Web App**.
- Use **SHAP (SHapley Additive Explanations)** for model interpretability.

## 📜 License
This project is for educational purposes only. Free to use and modify!

✨ Built with Python & Machine Learning ✨

