### showcase project: modeling and AI

### Info

This part of the project is created to showcase some data science & AI features.

## Table of Contents
1. [Setup](#setup)
2. [Services](#services)
3. [Folder Structure](#folder-structure)
4. [Documentation](#documentation)
5. [Dev Workflow](#dev-workflow)
6. [Branch Workflow](#branch-workflow)
7. [Design Choices](#design-choices)

## Setup

Run virtual environment, install dependencies and run the python code.  

```
python -m venv venv
source venv/Scripts/Activate

pip install -r requirements.txt

python ./src/main.py
```

Note: The above code snippet works on Windows/GitBash. Look up online documentation if your OS/Terminal/Shell is different.  

Related data files/folder will be created data/local  

If needed create the directories.  

## Services

- LLM Integration Framework - [Langchain](https://python.langchain.com/api_reference/reference.html)  

- Data Validation Tool - [Pydantic](https://docs.pydantic.dev/latest/)  

- Data Analysis Tool 1 - [Pandas](https://pandas.pydata.org/docs/)  

- Data Analysis Tool 2 - [Statsmodel](https://www.statsmodels.org/stable/api.html)  

## Folder Structure

./src: codebase (main.py, models.py, parser.py)  

./data/local: data files

## Documentation

- Project Board: Github Projects

## Dev Workflow

- ```git checkout main``` & ```git pull --rebase``` to stay updated with the latest updates.  
- Check out a new branch according to the branch workflow (eg: ```git checkout -b feat-1```).  
- ```pip freeze > requirements.txt``` to update packages docs
- Commit messages should be active rather than passive (eg: "Adds feature" as opposed to "feature Added").  
- Push changes and link branch to ticket.  
- Resolve conflict if it exists.  
- Repeat.

## Branch Workflow

Branch Name Convention:  
```
<type>-<ticket number from GitHub Projects>
```
- feat      - feature addition  
- docs      - documentation  
- fix       - bug fix  
- refactor  - improvements/tech debt  

## Design Choices

Python was used for this project due to its extensive data resources (forecasts, modelling etc.).  

Langchain is used as an AI integration framework due to available resources and friendly recommendations. It was easy to learn and implementation went with little hassle

Pydantic, like langchain, was another friendly recommendation. Data validation helped with recognizing bugs in the program throughout.

ARIMA model was used as a simple choice to show time series forecasting