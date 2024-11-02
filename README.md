# Aliyun Tianchi News Recommendation Competition Reproduction Code

This repository is used to document the code I reproduced from [datawhale](https://github.com/datawhalechina/fun-rec) during the Aliyun Tianchi news recommendation competition.

## Repository Structure

/project-root<br>
│<br>
├── /src    <br>               
│   ├── /data_analysis    <br>
│   │   └── data_analysis.ipynb<br>
│   ├── /metrics          <br> 
│   │   └── metrics_recall.py<br>
│   ├── /similarity      <br>  
│   │   ├── sim_matrix.py<br>
│   │   └── cold_start_items.py<br>
│   ├── /recall  <br>         
│   │   ├── recall.py<br>
│   │   ├── neg_sample.py<br>
│   │   └── sorting_with_model_ensemble.py<br>
│   ├── /feature_engineering <br>
│   │   └── feature_engineering.py<br>
│   ├── /data_reading  <br>    
│   │   └── data_reading.py<br>
│   ├── config.py   <br>      
│   ├── main.ipynb  <br>       
│   └── baseline.ipynb    <br>
│<br>
└── README.md    <br>        
<br>

### Contents of the src Folder

- `data_analysis.ipynb`: Data analysis notebook
- `metrics_recall.py`: Recall metric calculations
- `sim_matrix.py`: Similarity matrix computations
- `cold_start_items.py`: Handling cold start items
- `recall.py`: Recall logic
- `baseline.ipynb`: Baseline model
- `neg_sample.py`: Negative sampling
- `feature_engineering.py`: Feature engineering
- `data_reading.py`: Data reading
- `config.py`: Configuration file
- `sorting_with_model_ensemble.py`: Model ensemble sorting
- `main.ipynb`: Main implementation file

## Project Progress

The current progress of this project is **80%**, with the remaining tasks primarily focused on the implementation of the DIN model in the `main.ipynb` file and the model ensemble.

```plaintext
Progress Bar: [███████████████████░░░░░] 80%
