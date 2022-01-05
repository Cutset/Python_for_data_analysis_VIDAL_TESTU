from flask import Flask, render_template, request, redirect, session
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, IntegerField, SelectField, SubmitField
from wtforms.validators import DataRequired, InputRequired, NumberRange
import pandas as pd
#import ipynb.fs.defs.valeurs_foncieres # importe les fonctions et classes définies dans le notebook valeurs_foncieres (ne fonctionne pas avec les notebook colab visiblement)
import model_viz as mv
import pickle 

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import train_test_split
app = Flask(__name__)
Bootstrap(app=app)
#Menu(app=app)
app.config["SECRET_KEY"] = "hard to guess string"

data, data1 = None, None

data = pd.read_csv("data/OnlineNewsPopularity.csv")
data1 = pd.read_csv("data/data_modif.csv")

X = data1.drop([" shares", "url", " timedelta", "popular"], axis = 1)
y = data1["popular"]
X = X.drop(['Unnamed: 0'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
X_train_regressor, X_test_regressor, y_train_regressor, y_test_regressor = train_test_split(X, data1[" shares"], test_size = 0.2, random_state = 0)

##
## Forms
##

class ChooseModel(FlaskForm):
    model = [('models/rforest.sav', 'RandomForestClassifier'), ('models/adaboost.sav', 'AdaBoost'), ('models/knn.sav', 'KNN'), ('models/GNB.sav', 'GaussianNB')] #(valeur à renvoyer, valeur à afficher)
    arg1 = SelectField(
        "Choose model", 
        choices= model,
        validators=[InputRequired(model)], 
        coerce=str)
    viz = [('Classification Report', 'Classification Report'), ('Confusion matrix', 'Confusion matrix'), ('ROC curve', 'ROC curve')] #(valeur à renvoyer, valeur à afficher)
    arg2 = SelectField(
        "Choose what you want to display", 
        choices= viz,
        validators=[InputRequired(viz)], 
        coerce=str)
    submit = SubmitField("Display")

class ChooseModelRegressor(FlaskForm):
        model = [('models/ridge_regression.sav', 'Ridge Regression'), ('models/rforest_regressor.sav', 'Random Forest Regressor')] #(valeur à renvoyer, valeur à afficher)
        arg1 = SelectField(
            "Choose model", 
            choices= model,
            validators=[InputRequired(model)], 
            coerce=str)
        submit = SubmitField("Display")

##
## Routes
##
@app.route('/')
def index():
    if data is not None:
        if data1 is not None:
            session['data_loaded']= "Toutes les données sont importées"
        else: session['data_loaded']= "Les données 2020 sont déjà importées"
    else:
        session['data_loaded']= ""
    return render_template('base.html', data_loaded= session['data_loaded'])

@app.route('/load_data')
def load_data():
    global data1
    global data
    if data is not None and data1 is not None:
        return redirect('/')
    else:
        if data is None:
            donnees = pd.read_csv("https://www.data.gouv.fr/fr/datasets/r/90a98de0-f562-4328-aa16-fe0dd1dca60f", sep='|', decimal= ',')
            data = donnees[["Date mutation","Nature mutation","Valeur fonciere","Code voie","B/T/Q","Type de voie","Voie","Code postal","Commune","Code departement","Code commune","Section","No plan","Nombre de lots","Type local","Surface reelle bati","Nombre pieces principales","Nature culture","Nature culture speciale","Surface terrain"]]
        if data1 is None:
            donnees1 = pd.read_csv("https://www.data.gouv.fr/fr/datasets/r/3004168d-bec4-44d9-a781-ef16f41856a2", sep='|', decimal= ',')
            data1 = donnees1[["Date mutation","Nature mutation","Valeur fonciere","Code voie","B/T/Q","Type de voie","Voie","Code postal","Commune","Code departement","Code commune","Section","No plan","Nombre de lots","Type local","Surface reelle bati","Nombre pieces principales","Nature culture","Nature culture speciale","Surface terrain"]]
        return redirect('/') 

@app.route('/notebook')
def notebook():
    return render_template('notebook_.html')

@app.route('/model')
def model():
    global X_test
    global y_test
    model = pickle.load(open('models\\final_model.sav', 'rb'))
    cv = classification_report(y_test, model.predict(X_test), digits = 4, output_dict = True)
    cm = confusion_matrix(y_test, model.predict(X_test))
    roc = mv.roc(model, X_test, y_test)
    return render_template('model.html', cv = cv, cm = cm, roc=roc)

@app.route("/viz", methods=["GET", "POST"])
def Viz():
    global data1
    global X_test
    global y_test
    form= ChooseModel()
    image=""
    if request.method == "POST" and form.validate_on_submit():
        image = mv.viz(form.arg1.data, form.arg2.data, X_test, y_test) #form.arg.data représente la donnée (str) contenue dans form.arg (le choix de l'utilisateur)
    return render_template("form_plot_agg.html", image=image, form=form)

@app.route("/viz_regressor", methods=["GET", "POST"])
def Viz_regressor():
    global X_test_regressor
    global y_test_regressor
    form= ChooseModelRegressor()
    image=""
    if request.method == "POST" and form.validate_on_submit():
        model = pickle.load(open(form.arg1.data, 'rb'))
        image = f"""
        </hr>
        </hr>
        <h5>Accuracy :</h5>
        <p>{model.score(X_test_regressor, y_test_regressor)}</p>"""
    return render_template("form_plot_agg.html", image=image, form=form)

@app.route('/html_table', methods=["POST", "GET"])
def html_table():
    global data
    head= data.head(10)
    return render_template('base.html',  tables=[head.to_html(classes='data')], titles=head.columns.values)

@app.route("/plottest", methods=["GET"])
def plotView():
    image = vf.test_plot()
    return render_template("image.html", image=image)


if __name__ == "__main__":
    # run server
    print(app.url_map)
    app.run(host= "0.0.0.0", port= 5000, debug= True)