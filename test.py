from flask import Flask

#Create a Flask app instance
app = Flask(__Name__)

#Define a route and a view function
@app.route('/')
def hello():
    return'Hello, World'
#run the app if this is executed
if __name__ == '__main__':
    app.run(debug=True)