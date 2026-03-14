import joblib
import numpy as np
from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI()

# Load ML model
model = joblib.load("Wine_quality.joblib")


# ---------------- HOME PAGE ---------------- #

@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!DOCTYPE html>
<html>
<head>

<title>AI Wine Intelligence</title>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>

*{
margin:0;
padding:0;
box-sizing:border-box;
font-family:'Poppins',sans-serif;
}

body{
height:100vh;
display:flex;
justify-content:center;
align-items:center;
background:linear-gradient(135deg,#2c3e50,#4ca1af);
}

/* Card */

.container{
width:700px;
padding:40px;
border-radius:20px;
background:rgba(255,255,255,0.1);
backdrop-filter:blur(15px);
box-shadow:0 10px 40px rgba(0,0,0,0.4);
color:white;
}

/* Title */

.title{
text-align:center;
margin-bottom:30px;
}

.title h1{
font-size:32px;
}

.title p{
font-size:14px;
opacity:0.8;
}

/* Form Grid */

.form-grid{
display:grid;
grid-template-columns:1fr 1fr;
gap:15px;
}

.input-box{
display:flex;
flex-direction:column;
}

label{
font-size:13px;
margin-bottom:5px;
}

input{
padding:10px;
border-radius:8px;
border:none;
outline:none;
}

/* Button */

.predict-btn{
margin-top:25px;
width:100%;
padding:14px;
border:none;
border-radius:25px;
background:linear-gradient(90deg,#ff512f,#dd2476);
color:white;
font-size:16px;
cursor:pointer;
transition:0.3s;
}

.predict-btn:hover{
transform:scale(1.05);
box-shadow:0 0 15px rgba(255,80,80,0.7);
}

.footer{
margin-top:20px;
text-align:center;
font-size:12px;
opacity:0.7;
}

</style>
</head>

<body>

<div class="container">

<div class="title">
<h1>🍷 AI Wine Intelligence</h1>
<p>Predict Wine Quality Using Machine Learning</p>
</div>

<form action="/predict" method="post">

<div class="form-grid">

<div class="input-box">
<label>Density</label>
<input type="number" step="any" name="density" required>
</div>

<div class="input-box">
<label>Alcohol</label>
<input type="number" step="any" name="alcohol" required>
</div>

<div class="input-box">
<label>Fixed Acidity</label>
<input type="number" step="any" name="fixed_acidity" required>
</div>

<div class="input-box">
<label>Sulphates</label>
<input type="number" step="any" name="sulphates" required>
</div>

<div class="input-box">
<label>Volatile Acidity</label>
<input type="number" step="any" name="volatile_acidity" required>
</div>

<div class="input-box">
<label>Citric Acid</label>
<input type="number" step="any" name="citric_acid" required>
</div>

<div class="input-box">
<label>Total Sulfur Dioxide</label>
<input type="number" step="any" name="total_sulfur_dioxide" required>
</div>

</div>

<button class="predict-btn">🚀 Predict Wine Quality</button>

</form>

<div class="footer">
Powered by FastAPI + Machine Learning
</div>

</div>

</body>
</html>
"""


# ---------------- PREDICTION ---------------- #

@app.post("/predict", response_class=HTMLResponse)
def predict(
    density: float = Form(...),
    alcohol: float = Form(...),
    fixed_acidity: float = Form(...),
    sulphates: float = Form(...),
    volatile_acidity: float = Form(...),
    citric_acid: float = Form(...),
    total_sulfur_dioxide: float = Form(...)
):

    # Correct feature order
    features = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        density,
        sulphates,
        alcohol,
        total_sulfur_dioxide
    ]])

    prediction = model.predict(features)[0]

    # Emoji logic
    if prediction >= 7:
        emoji = "👍"
        message = "Excellent Wine Quality"
    elif prediction >= 5:
        emoji = "🙂"
        message = "Average Wine Quality"
    else:
        emoji = "👎"
        message = "Poor Wine Quality"


    return f"""
<html>

<head>

<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600&display=swap" rel="stylesheet">

<style>

body {{
display:flex;
justify-content:center;
align-items:center;
height:100vh;
background:linear-gradient(135deg,#1f4037,#99f2c8);
font-family:Poppins;
color:white;
}}

.card {{
background:rgba(0,0,0,0.4);
padding:50px;
border-radius:20px;
text-align:center;
box-shadow:0 10px 40px rgba(0,0,0,0.4);
}}

h1 {{
font-size:55px;
margin-top:10px;
}}

.emoji {{
font-size:80px;
margin-top:20px;
}}

.message {{
font-size:20px;
margin-top:10px;
}}

a {{
display:inline-block;
margin-top:25px;
padding:12px 25px;
background:#ff4b2b;
color:white;
border-radius:20px;
text-decoration:none;
}}

</style>

</head>

<body>

<div class="card">

<h2>🍷 Predicted Wine Quality</h2>

<h1>{prediction}</h1>

<div class="emoji">{emoji}</div>

<div class="message">{message}</div>

<a href="/">Analyze Another</a>

</div>

</body>

</html>
"""


# ---------------- RUN SERVER ---------------- #

if __name__ == "__main__":
    uvicorn.run("wine_app:app", host="0.0.0.0", port=8000, reload=True)