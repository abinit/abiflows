#!/usr/bin/env python
# coding: utf-8
"""Flask application."""
from __future__ import print_function, division, unicode_literals

from flask import Flask
from flask.ext.mongoengine import MongoEngine

from abiflows.core.models import *

from mongoengine import connect
connect('abiflows')

db = MongoEngine()

app = Flask(__name__)
app.config.from_pyfile('config.py')
db.init_app(app)


# Paginate through todo
def view_todos(page=1):
    #paginated_todos = MongoFlow.objects.paginate(page=1, per_page=10)
    lines = []
    for flow in MongoFlow.objects:
        lines.append(str(flow))

    return "\n".join(lines)

@app.route("/")
def hello():
    lines = []
    app = lines.append
    app("database: %s" % str(db))
    app("Number of flows: %s" % MongoFlow.objects.count())

    return "\n".join(lines)
    #return "Hello World!"
    #return view_todos()


if __name__ == "__main__":
    app.run(debug=True)

