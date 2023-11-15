import json
from query import query_joplin
from flask import Flask, jsonify, render_template, request, Response

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search')
def search():
    query = request.args.get('q')

    df = query_joplin(query)
    results = df.to_html(index=False, escape=False)
    return jsonify({'result': results})

if __name__ == '__main__':
    app.run(debug=True)
