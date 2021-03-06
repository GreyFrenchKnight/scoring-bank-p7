# scoring-bank-p7
API of P7 Scoring Bank

![ml_life_cycle](https://github.com/incubated-geek-cc/mental-healthcare-predictors/blob/main/ml_life_cycle.png)

https://towardsdatascience.com/deploy-your-python-machine-learning-models-on-heroku-in-3-steps-dc5b6aca73d9

#### API deployed at: https://scoring-bank-p7.herokuapp.com/

#### Step 1. Create a new app on Heroku "scoring-bank-p7"

#### Step 2. Initiate Project Folder on Local PC
```
[any folder pathname] > cd [project folder pathname]
[project folder pathname] > python -m venv venv
[project folder pathname] > CALL venv/Scripts/activate.bat
[project folder pathname] > python -m pip install --upgrade pip
[project folder pathname] > pip install numpy pandas joblib pydantic fastapi sklearn uvicorn gunicorn
[project folder pathname] > pip freeze > requirements.txt
```

Develop the functionnalities inside a python file.
```
[project folder pathname] > uvicorn app:app
```

#### Step 3. Initiate GIT in the project folder
Test before commit.
Save to Git :
```
[project folder pathname] > git init
[project folder pathname] > git add .
[project folder pathname] > git commit -m "first commit"
```

Deploy on Heroku :
```
[project folder pathname] > heroku login
[project folder pathname] > heroku git:remote -a scoring-bank-p7
[project folder pathname] > git push heroku master
```

#### Step 4. Testing API
* Heroku dashboard https://dashboard.heroku.com/apps/scoring-bank-p7
* Accessing app https://scoring-bank-p7.herokuapp.com/
* Checking logs https://dashboard.heroku.com/apps/scoring-bank-p7/logs