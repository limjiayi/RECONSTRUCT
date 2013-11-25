from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Table, Column, Integer, String, Float, Date, ForeignKey, MetaData
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, backref

Base = declarative_base()

ENGINE = create_engine("postgresql+psycopg2://user:user@localhost:5432/user", echo=False)
Session = sessionmaker(bind=ENGINE)
session = Session()
metadata = MetaData()
metadata.bind = ENGINE

cameras_table = Table('cameras', metadata,
               Column('id', Integer, primary_key=True),
               Column('model', String),
               Column('sensor_width', Float))

cameras_table.create(checkfirst=True)

class Camera(Base):
  __tablename__ = 'cameras'

  id = Column(Integer, primary_key=True)
  model = Column(String(64), nullable=False)
  sensor_width = Column(Float, nullable=False)

def add_camera(model, sensor_width):
  new_camera = Camera(model=model.upper(), sensor_width=sensor_width)
  session.add(new_camera)
  session.commit()
  session.refresh(new_camera)

def remove_camera(model):
  camera = session.query(Camera).filter_by(model=model.upper()).one()
  session.delete(camera)
  session.commit()

def get_sensor_size(model):
  rows = session.query(Camera).filter_by(model=model.upper()).one()
  return rows.sensor_width