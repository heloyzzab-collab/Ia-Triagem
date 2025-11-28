from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import joblib

app = Flask(__name__)
app.secret_key = "mahoraga" 

modelo = joblib.load("modelo.pkl")

ADMIN_USER = "admin"
ADMIN_PASS = "sukuna"

#rotas
@app.route('/')
def home():
    return redirect("/triagem")
@app.route("/triagem")
def triagem():
    return render_template("triagem.html")
@app.route("/resultado", methods=["POST"])
def resultado():
    dados = {
        "sintomas": [request.form["sintomas"]],
        "idade": [int(request.form["idade"])],
        "temperatura": [float(request.form["temperatura"])],
        "doenca_cronica": [request.form["doenca"]],
        "tempo_sintomas_h": [float(request.form["tempo"])],
        "nivel_dor": [int(request.form["dor"])],
        "consegue_mover": [request.form["mover"]]
    }
    df = pd.DataFrame(dados)
    risco_previsto = modelo.predict(df)[0]
    df["risco_previsto"] = risco_previsto
    if os.path.exists("dataSet/novosCases.csv"):
        antigo = pd.read_csv("dataSet/novosCases.csv", sep=";")
        novo = pd.concat([antigo, df], ignore_index=True)
        novo.to_csv("dataSet/novosCases.csv", sep=";", index=False)
    else: 
        df.to_csv("dataSet/novosCases.csv", sep=";", index=False)
    return render_template("resultado.html", risco=risco_previsto)
    
#rotaADM

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = request.form["usuario"]
        senha = request.form["senha"]
        
        if user == ADMIN_USER and senha == ADMIN_PASS:
            session["admin"] = True
            return redirect("/admin")
        else:
            return "Credenciais inválidas!"
        
    return render_template("login.html")

@app.route("/admin")
def admin():
    if "admin" not in session:
        return redirect("/login")
    if not os.path.exists("dataSet/novosCases.csv"):
        tabela = pd.DataFrame()
    else:
        tabela = pd.read_csv("dataSet/novosCases.csv", sep=";")
    return render_template("admin.html", tabela=tabela.to_dict(orient="records"))

@app.route("/admin/corrigir/<int:indice>/<risco>")
def corrigir(indice, risco):
    if "admin" not in session:
        return redirect("/login")
    
    df = pd.read_csv("dataSet/novosCases.csv", sep=";")
    
    caso = df.iloc[[indice]].copy()
    caso["risco_correto"] = risco
    
    if not os.path.exists("dataSet/casesCorrigidos.csv"):
        caso.to_csv("dataSet/casesCorrigidos.csv", sep=";", index=False)
    else: 
        antigo = pd.read_csv("dataSet/casesCorrigidos.csv", sep=";")
        novo = pd.concat([antigo, caso], ignore_index=True)
        novo.to_csv("dataSet/casesCorrigidos.csv", sep=";", index=False)
        
    df = df.drop(indice)
    df.to_csv("dataSet/novosCases.csv", sep=";", index=False)
    
    return redirect("/admin")

@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect("/login")

if __name__ == "__main__":
    app.run(debug=True)