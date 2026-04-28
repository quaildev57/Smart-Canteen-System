# Smart Canteen System: Food Demand and Wastage Prediction

## Overview

This project focuses on analyzing canteen operations data to improve decision-making through data-driven insights. It aims to predict food demand, understand consumption patterns, and minimize food wastage using data analysis and machine learning techniques.

## Objectives

* Forecast daily food demand
* Identify patterns in customer behavior
* Analyze and reduce food wastage
* Provide actionable insights through visualizations

## Project Structure

```
smart_canteen_fda/
│
├── 1_setup.py        # Data loading and preprocessing
├── 2_eda.py          # Exploratory data analysis and visualization
├── 3_model.py        # Machine learning model for demand prediction
├── 4_wastage.py      # Food wastage analysis
├── app.py            # Main application entry point
├── canteen_clean.csv # Processed dataset
└── charts/           # Generated visualizations
```

## Methodology

### Data Preprocessing

* Cleaning and formatting raw data
* Handling missing values
* Feature preparation for analysis and modeling

### Exploratory Data Analysis

* Trend analysis of food demand
* Identification of peak hours and popular items
* Visualization of key patterns

### Model Development

* Training machine learning models for demand prediction
* Evaluating model performance
* Generating predictions for decision support

### Wastage Analysis

* Measuring excess food production
* Identifying high-wastage items
* Recommending optimization strategies

## Installation

1. Clone the repository:

```
git clone https://github.com/your-username/your-repo-name.git
```

2. Navigate to the project directory:

```
cd your-repo-name
```

3. Install required dependencies:

```
pip install -r requirements.txt
```

## Usage

Run the scripts in the following order:

```
python 1_setup.py
python 2_eda.py
python 3_model.py
python 4_wastage.py
python app.py
```

## Output

* Analytical charts stored in the `charts/` directory
* Model predictions for food demand
* Insights into wastage patterns

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib / Seaborn
* Scikit-learn

## Future Improvements

* Deploy as a web-based dashboard
* Integrate real-time data collection
* Enhance model accuracy with additional features
* Add user interface for canteen staff
