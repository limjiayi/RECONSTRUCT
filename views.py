from flask import Flask, render_template, redirect, request, session, url_for, flash, jsonify
from sqlalchemy import desc
import os
import config
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

@app.route('/upload', methods=['POST'])
def upload():
    username = session.get('username')
    if username:
        user_id = model_session.query(model.User).filter_by(username=username).first().id
    else:
        user_id = 0

    f_keys = request.form.keys()
    pattern = re.compile(r'^data:image/(png|jpeg|jpg);base64,(.*)$')
    raw_data = []
    for key in f_keys:
        if key != 'cloud_name':
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

    cloud_name = request.form['cloud_name']

    # save cloud to database
    new_cloud = model.Cloud(user_id=user_id, name=cloud_name)
    model_session.add(new_cloud)
    model_session.commit()
    model_session.refresh(new_cloud)

    cloud_id = model_session.query(model.Cloud.id).order_by(desc(model.Cloud.id)).first()[0]

    # create a new directory inside the user's directory for uploaded photos
    path = user_path + '/' + str(cloud_id)
    if not os.path.exists(path):
        os.mkdir(path)

    for idx, d in enumerate(raw_data):
        filename = '{}.{}'.format(len(raw_data)-idx, match.group(1))
        with open(path + '/' + filename, 'wb') as f:
            f.write(d)

    path = 'static/uploads/%d/%s' % (user_id, cloud_id)

    images = sorted(path + '/' + img for img in os.listdir(path) if img.rpartition('.')[2].lower() in ('jpg', 'jpeg', 'png'))
    points_path = os.path.abspath(os.path.join(path, "points"))
    reconstruct.start(images, points_path)
    points = str(reconstruct.extract_points(points_path + ".txt"))

    # set the path to the text file storing the 3D points of the cloud
    cloud = model_session.query(model.Cloud).filter_by(id=cloud_id).first()
    cloud.path = path
    model_session.commit()

    # save photos to database
    photos = [ img for img in os.listdir(path) if img.rpartition('.')[2].lower() in ('jpg', 'jpeg', 'png') ]

    for photo in photos:
        new_photo = model.Photo(filename=photo, path=path, cloud_id=cloud_id)
        model_session.add(new_photo)
        model_session.commit()
        model_session.refresh(new_photo)

    return points

@app.route('/cloud/<id>')
def get_cloud(id):
    '''Gets the 3D points for a past point cloud.'''
    cloud_id = id
    path = model_session.query(model.Cloud).filter_by(id=cloud_id).first().path
    points = str(reconstruct.extract_points(path + '/points.txt'))
    return points

@app.route('/past/<user_id>')
def get_past_clouds(user_id):
    '''Gets the name and photos used to reconstruct all of the user's past point clouds.'''
    if user_id:
        clouds = model_session.query(model.User).filter_by(id=user_id).first().clouds
        if clouds != "":
            clouds_d = dict( (cloud.name, [ photo.to_dict() for photo in cloud.photos ]) for cloud in clouds )
            return jsonify(clouds_d)
    return None

@app.route('/past/<user_id>/<cloud_id>', methods=['POST'])
def remove_cloud(cloud_id):
    if user_id:
        cloud = model_session.query('clouds').filter_by(cloud_id=cloud_id).one()
        # best effort delete: try to delete a file if it exists, don't delete if lack permissions
        try:
            os.remove(cloud.path)
        except OSError:
            pass
        for photo in cloud.photos:
            model.delete_photo(photo.id)
        model.delete_cloud(cloud_id)

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
        try:
            user = model_session.query(model.User).filter_by(username=username).one()
            if user.authenticate(password):
                session['username'] = username
                return redirect(url_for('display_user', username=username))
            else:
                flash('The username or password is incorrect. Please try again.')
                return redirect(url_for('login'))
        except:
            flash('This user does not exist.')
            return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('index'))

@app.route('/<username>')
def display_user(username):
    username = session.get('username')
    if username:
        try:
            user_id = model_session.query(model.User).filter_by(username=username).first().id
            clouds = model_session.query(model.User).filter_by(id=user_id).first().clouds
            return render_template('display_user.html', username=username, user_id=user_id, clouds=clouds)
        except:
            return redirect(url_for('login'))

if __name__ == "__main__":
    app.run(debug=True)