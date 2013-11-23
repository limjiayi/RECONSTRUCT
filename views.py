from flask import Flask, render_template, redirect, request, g, session, url_for, flash, jsonify
from sqlalchemy import desc
import os
import config
import forms
import model
from model import session as model_session
import reconstruct
import base64
import re

app = Flask(__name__)
app.config.from_object(config)


@app.route('/')
def index():
    username = session.get('username')
    if username:
        return redirect(url_for('display_user', username=username))
    else:
        return render_template('index.html', username=username)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    username = session.get('username')
    if username:
        user_id = model_session.query(model.User).filter_by(username=username).first().id
    else:
        user_id = 0

    if request.method == 'POST':
        f_keys = request.form.keys()
        pattern = re.compile(r'^data:image/(png|jpeg|jpg);base64,(.*)$')
        raw_data = []
        for key in f_keys:
            match = pattern.match(request.form[key])
            if match is None:
                raise ValueError('Invalid image data.')

            content_type = 'image/{}'.format(match.group(1))
            file_data = base64.b64decode(match.group(2))
            raw_data.append(file_data)

        user_path = 'static/uploads/%d' % user_id
        if not os.path.exists(user_path):
            # user doesn't have his/her own directory yet, so create one
            os.mkdir(user_path)
        folders = os.listdir(user_path)
        if folders == []:
            last_id = 0
        else:
            last_id = max(folders)
        path = user_path + '/' + str(int(last_id)+1)

        # create a new directory inside the user's directory for uploaded photos
        if not os.path.exists(path):
            os.mkdir(path)
        for idx, d in enumerate(raw_data):
            filename = '{}.{}'.format(len(raw_data)-idx, match.group(1))
            with open(path + '/' + filename, 'wb') as f:
                f.write(d)

        folders = os.listdir('static/uploads/%d' % user_id)
        last_id = max(folders)
        path = 'static/uploads/%d/%s' % (user_id, last_id)
        print path + '/' + os.listdir(path)[0]

        images = sorted(path + '/' + img for img in os.listdir(path) if img.rpartition('.')[2].lower() in ('jpg', 'jpeg', 'png'))
        print "images: ", images
        points_path = os.path.abspath(os.path.join(path, "points"))
        os.mkdir(points_path)
        print "points path: ", points_path
        reconstruct.start(images, points_path)

        # save cloud to database
        new_cloud = model.Cloud(user_id=user_id, path=path)
        model_session.add(new_cloud)
        model_session.commit()
        model_session.refresh(new_cloud)

        cloud_id = model_session.query(model.Cloud.id).filter_by(user_id=user_id).\
                                order_by(desc(model.Cloud.id)).first()

        # save photos to database
        photos = os.listdir(path)

        for photo in photos:
            new_photo = model.Photo(filename=photo, path=path, cloud_id=cloud_id)
            model_session.add(new_photo)
            model_session.commit()
            model_session.refresh(new_photo)

    elif request.method == 'GET':
        points = reconstruct.extract_points(points_path + ".txt")
        return points


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'GET':
        return render_template('register.html')
    else:
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

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    else:
        username = request.form.get('username')
        password = request.form.get('password')
        user = model_session.query(model.User).filter_by(username=username).one()
        if user.authenticate(password):
            session['username'] = username
            return redirect(url_for('display_user', username=username))
        else:
            flash('The username or password is incorrect. Please try again.')
            return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/<username>')
def display_user(username):
    username = session.get('username')
    clouds = model.query(model.User).filter_by(username=username).first().clouds
    return render_template('display_user.html', username=username, clouds=clouds)


if __name__ == "__main__":
    app.run(debug=True)