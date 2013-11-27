import config
import bcrypt
from datetime import datetime
import os

from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, ForeignKey
from sqlalchemy import Column, Integer, String, DateTime, Text, LargeBinary
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, backref
from flask.ext.login import UserMixin

ENGINE = create_engine(config.DB_URI, echo=False) 
session = scoped_session(sessionmaker(bind=ENGINE, autocommit=False, autoflush=False))

Base = declarative_base()
Base.query = session.query_property()

### Class declarations ###

class User(Base, UserMixin):
    __tablename__ = 'users'
    __table_args__ = {'sqlite_autoincrement': True}

    id = Column(Integer, primary_key=True)
    username = Column(String(64), nullable=False)
    password = Column(String(64), nullable=False)
    salt = Column(String(64), nullable=False)
    clouds = relationship('Cloud', backref=backref('author', uselist=True)) # user.clouds, cloud.author

    def set_password(self, password):
        self.salt = bcrypt.gensalt()
        password = password.encode("utf-8")
        self.password = bcrypt.hashpw(password, self.salt)

    def authenticate(self, password):
        password = password.encode("utf-8")
        return bcrypt.hashpw(password, self.salt.encode("utf-8")) == self.password

    def get_id(self):
        return unicode(self.id)

    def to_dict(self):
        return dict((c.name, getattr(self, c.name)) for c in self.__table__.columns)

class Cloud(Base):
    __tablename__ = 'clouds'
    __table_args__ = {'sqlite_autoincrement': True}

    id = Column(Integer, primary_key=True)
    name = Column(String(64), nullable=True)
    path = Column(String(256), nullable=True)
    created_on = Column(DateTime, nullable=False, default=datetime.now)
    user_id = Column(Integer, ForeignKey('users.id'), nullable=False)
    photos = relationship('Photo', backref=backref('cloud', uselist=True)) # cloud.photos, photo.cloud

    def to_dict(self):
        return dict((c.name, str(getattr(self, c.name))) for c in self.__table__.columns)

class Photo(Base):
    __tablename__ = 'photos'
    __table_args__ = {'sqlite_autoincrement': True}

    id = Column(Integer, primary_key=True)
    filename = Column(String(64), nullable=False)
    path = Column(String(256), nullable=False)
    uploaded_on = Column(DateTime, nullable=False, default=datetime.now)
    cloud_id = Column(Integer, ForeignKey('clouds.id'), nullable=False)

    def to_dict(self):
        return dict((c.name, str(getattr(self, c.name))) for c in self.__table__.columns)

### End of class declarations ###

def delete_user(username):
    session.query(User).filter(User.username==username).delete()
    session.commit()

def delete_cloud(cloud_id):
    session.query(Cloud).filter(Cloud.id==cloud_id).delete()
    session.commit()

def delete_photos(cloud_id):
    session.query(Photo).filter(Photo.cloud_id==cloud_id).delete()
    session.commit()


def main():
    Base.metadata.create_all(ENGINE)

if __name__ == '__main__':
    main()