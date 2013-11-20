from flask import Flask, render_template, redirect, request, g, session, url_for, flash
import os
import config
import forms
import model
from model import session as model_session

app = Flask(__name__)
app.config.from_object(config)


@app.route('/')
def index():
    username = session.get('username')
    photos = os.listdir('static/uploads')
    if username:
        return redirect(url_for('display_user'))
    else:
        return render_template('index.html', photos=photos)

@app.route('/register')
def register():
    return render_template('register.html')

@app.route('/register', methods=['POST'])
def create_account():
    username = request.form.get('username')
    user = model.User.query.filter_by(username=username).first()
    if user != None:
        flash('This username is already taken.')
        return redirect(url_for('register'))
    else:
        password = request.form.get('password')
        verify_password = request.form.get('verify_password')

        if verify_password == password:
            new_user = model.User(username=username)
            new_user.set_password(password)
            model_session.add(new_user)
            model_session.commit()
            model_session.refresh(new_user)
            return redirect(url_for('login'))
        else:
            flash('Passwords do not match!')
            return redirect(url_for('register'))

@app.route('/login')
def login():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def sign_in():
    username = request.form.get('username')
    password = request.form.get('password')
    user = model_session.query(model.User).filter_by(username=username).one()
    if user.authenticate(user.password) == password:
        session['username'] = username
        return redirect(url_for('display_user'))
    else:
        flash('The username or password is incorrect. Please try again.')
        return redirect(url_for('login'))


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/<username>')
def display_user():
    username = session.get('username')
    return render_template('display_user.html', username=username)

@app.route('/', methods=['GET', 'POST'])
def get_uploaded_files():
    if request.method == 'POST':
        photos = request.json()
        for photo in photos:
            model.add_photo(photo.filename)
        return redirect(url_for('index'))

### functions ###


if __name__ == "__main__":
    app.run(debug=True)