from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Date, ForeignKey
from sqlalchemy.orm import sessionmaker, scoped_session, relationship, backref

ENGINE = create_engine("sqlite:///cameras.db", echo=False)
db_session = scoped_session(sessionmaker(bind=ENGINE, autocommit=False, autoflush=False))
Base = declarative_base()
Base.query = db_session.query_property()

class Cameras(Base):
    __tablename__ = "Cameras"

    id = Column(Integer, primary_key=True)
    model = Column(String(64), nullable=False)
    sensor_width = Column(String(5), nullable=False)

def add_camera(model, sensor_width):
    new_camera = Cameras(model=model, sensor_width=sensor_width)
    db_session.add(new_camera)
    db_session.commit()
    db_session.refresh(new_camera)

def get_sensor_size(model):
    sensor_width = db_session.query(Cameras).filter_by(model=model)
    return sensor_width

def main():
    pass

if __name__ == "__main__":
    main()