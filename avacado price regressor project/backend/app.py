from flask import Flask, render_template, request, jsonify
import pandas as pd
import pickle
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

# Get the base directory (parent of backend folder)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

app = Flask(__name__)

# List of models based on your training script[cite: 2]
model_names = [
    'LinearRegression', 'RobustRegression', 'RidgeRegression', 'LassoRegression', 'ElasticNet',
    'PolynomialRegression', 'SGDRegressor', 'ANN', 'RandomForest', 'SVM', 'LGBM',
    'XGBoost', 'KNN'
]

# Load models from the backend directory
models = {}
for name in model_names:
    model_path = f'backend/{name}.pkl'
    if os.path.exists(model_path):
        models[name] = pickle.load(open(model_path, 'rb'))

# Load evaluation results generated during training
try:
    results_df = pd.read_csv(os.path.join(BASE_DIR, 'model_evaluation_results.csv'))
except FileNotFoundError:
    results_df = pd.DataFrame()

@app.route('/')
def index():
    # Compute the four summary graphs and answers to show on the frontend
    csv_path = os.path.join(BASE_DIR, 'avocado.csv')
    
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return f"Error loading data: {e}", 500

    # prepare data similar to notebook
    def get_average_between(df, col1, col2):
        vals = []
        groups = list(df[col1].unique())
        for g in groups:
            sub = df[df[col1] == g]
            vals.append(sub[col2].mean())
        res = pd.DataFrame({'column1': groups, 'column2': vals})
        res = res.sort_values('column2', ascending=False).reset_index(drop=True)
        return res

    # 1) Price by region
    price_by_region = get_average_between(df, 'region', 'AveragePrice')
    highest_region = price_by_region['column1'].iloc[0]
    lowest_region = price_by_region['column1'].iloc[-1]

    # plot price by region
    images_dir = os.path.join(os.getcwd(), 'static', 'images')
    os.makedirs(images_dir, exist_ok=True)
    p1_path = os.path.join(images_dir, 'price_by_region.png')
    plt.figure(figsize=(12, 4))
    sns.barplot(x='column1', y='column2', data=price_by_region, palette='rocket')
    plt.xticks(rotation=90)
    plt.title('Average Price by Region')
    plt.tight_layout()
    plt.savefig(p1_path)
    plt.close()
    print(f"Saved image to: {p1_path}")

    # 2) Production (Total Volume) by region
    volume_by_region = get_average_between(df, 'region', 'Total Volume')
    highest_prod_region = volume_by_region['column1'].iloc[0]
    p2_path = os.path.join(images_dir, 'production_by_region.png')
    plt.figure(figsize=(12, 4))
    sns.barplot(x='column1', y='column2', data=volume_by_region, palette='viridis')
    plt.xticks(rotation=90)
    plt.title('Average Total Volume by Region')
    plt.tight_layout()
    plt.savefig(p2_path)
    plt.close()
    print(f"Saved image to: {p2_path}")

    # 3) Average price by year
    price_by_year = get_average_between(df, 'year', 'AveragePrice')
    p3_path = os.path.join(images_dir, 'price_by_year.png')
    plt.figure(figsize=(8, 4))
    sns.barplot(x='column1', y='column2', data=price_by_year.sort_values('column1'), palette='magma')
    plt.title('Average Price by Year')
    plt.tight_layout()
    plt.savefig(p3_path)
    plt.close()
    print(f"Saved image to: {p3_path}")

    # 4) Average volume by year
    volume_by_year = get_average_between(df, 'year', 'Total Volume')
    p4_path = os.path.join(images_dir, 'volume_by_year.png')
    plt.figure(figsize=(8, 4))
    sns.barplot(x='column1', y='column2', data=volume_by_year.sort_values('column1'), palette='cubehelix')
    plt.title('Average Total Volume by Year')
    plt.tight_layout()
    plt.savefig(p4_path)
    plt.close()
    print(f"Saved image to: {p4_path}")

    answers = {
        'highest_price_region': str(highest_region),
        'lowest_price_region': str(lowest_region),
        'highest_production_region': str(highest_prod_region)
    }

    images = {
        'price_by_region': '/static/images/price_by_region.png',
        'production_by_region': '/static/images/production_by_region.png',
        'price_by_year': '/static/images/price_by_year.png',
        'volume_by_year': '/static/images/volume_by_year.png'
    }

    return render_template('index.html', model_names=model_names, answers=answers, images=images)

@app.route('/predict', methods=['POST'])
def predict():
    model_name = request.form['model']
    
    # Map form inputs to avocado dataset features[cite: 1, 3]
    input_data = {
        'Total Volume': float(request.form['Total Volume']),
        '4046': float(request.form.get('4046', 0)),
        '4225': float(request.form.get('4225', 0)),
        '4770': float(request.form.get('4770', 0)),
        'Total Bags': float(request.form['Total Bags']),
        'Small Bags': float(request.form.get('Small Bags', 0)),
        'Large Bags': float(request.form.get('Large Bags', 0)),
        'XLarge Bags': float(request.form.get('XLarge Bags', 0)),
        'type': int(request.form['type']), # 0 for conventional, 1 for organic
        'year': int(request.form['year']),
        'region': int(request.form['region']) # Encoded integer[cite: 1]
    }
    
    input_df = pd.DataFrame([input_data])
    
    if model_name in models:
        model = models[model_name]
        prediction = model.predict(input_df)[0]
        # Ensure prediction is a standard float for rendering
        predicted_val = round(float(prediction), 2)
        return render_template('results.html', prediction=predicted_val, model_name=model_name)
    else:
        return jsonify({'error': 'Model not found'}), 400

@app.route('/results')
def view_results():
    # Display the performance table from the CSV
    return render_template('model.html', tables=[results_df.to_html(classes='data')], titles=results_df.columns.values)

if __name__ == '__main__':
    app.run(debug=True)
    

