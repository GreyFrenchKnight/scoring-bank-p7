# your-custom-app-name
#### What this is - A demonstration/tutorial on deploying a python Machine Learning model to Heroku via Flask 
![ml_life_cycle](https://github.com/incubated-geek-cc/mental-healthcare-predictors/blob/main/ml_life_cycle.png)
#### Full article can be found at: [Towards Data Science](https://towardsdatascience.com/deploy-your-python-machine-learning-models-on-heroku-in-3-steps-dc5b6aca73d9)
#### Web application deployed at: https://your-custom-app-name.herokuapp.com/

* **Step 0a.** Create virtual environment by running: `virtualenv .env` 
* **Step 0b.** Double-click the `pip_install_requirements.bat` to install all required python packages stated in the `requirements.txt` file into the virtual environment created

Step 1. Create a new app on Heroku "scoring-bank-p7"

Step 2. Initiate Project Folder on Local PC
Then test before commit
* >cd C:/Users/disch/Documents/OpenClassrooms/Workspace/20220411-Projet_7_Implementez_un_modele_de_scoring/Projet_7/repository/scoring-bank-p7
* >python -m venv venv
* >CALL venv/Scripts/activate.bat
* >pip install numpy pandas joblib pydantic fastapi sklearn lightgbm uvicorn gunicorn
* >pip freeze > requirements.txt

Step 3. Initiate GIT in the project folder
* > cd C:/Users/disch/Documents/OpenClassrooms/Workspace/20220411-Projet_7_Implementez_un_modele_de_scoring/Projet_7/repository/scoring-bank-p7
* [project folder pathname] > git init
* [project folder pathname] > git add .
* [project folder pathname] > git commit -m "first commit"
* [project folder pathname] > heroku login
* [project folder pathname] > heroku git:remote -a scoring-bank-p7
* [project folder pathname] > git push heroku master