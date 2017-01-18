import json
from flask import Flask, request, render_template, redirect
from math import floor
from urlparse import urlparse
import base64
import config as cfg
import sqlite3
import utils

app = Flask(__name__)

host = cfg.host
print host


def check_table_presence():
    create_table_query = "CREATE TABLE URL_DB(ID INTEGER PRIMARY KEY AUTOINCREMENT, URL TEXT NOT NULL, URLENCODED TEXT NOT NULL);"

    with sqlite3.connect('urls.db') as conn:
        cursor = conn.cursor()
        try:
            cursor.execute(create_table_query)
        except Exception as err:
            pass


@app.route('/', methods=['GET', 'POST'])
def home_screen():
    if request.method == 'POST':
        user_entered_url = request.form.get('url')
        if urlparse(user_entered_url).scheme == '':
            url = 'http://' + user_entered_url
        else:
            url = user_entered_url
        with sqlite3.connect('urls.db') as conn:
            cursor = conn.cursor()
            encoded_string = base64.urlsafe_b64encode(url)
            result = cursor.execute("INSERT INTO URL_DB (URL, URLENCODED) VALUES (?, ?)",
                                    [url, encoded_string])  # long value id is returned after the insert statement
            base62_encoded_string = utils.convert_to_base62(result.lastrowid)
        shortened_url = 'inception-' + encoded_string
        return render_template('home.html', short_url=shortened_url)
    return render_template('home.html')


@app.route('/<short_url>')
def redirecting_shortened_url(short_url):
    redirect_url = 'http://localhost:1723'
    encoded_part = short_url.split('inception-')[1]

    with sqlite3.connect('urls.db') as conn:
        cursor = conn.cursor()
        result = cursor.execute("SELECT URL FROM URL_DB WHERE URLENCODED = ?", [encoded_part])
        try:
            redirect_url = result.fetchone()[0]
        except Exception as err:
            print err
    return redirect(redirect_url)


if __name__ == '__main__':
    check_table_presence()
    app.run(host='0.0.0.0',
            debug=False, port=1723)
