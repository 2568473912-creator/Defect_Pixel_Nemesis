from sqlalchemy import create_engine, Column, Integer, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from datetime import datetime
import os
from utils.helpers import BASE_DIR # 引用工具层

Base = declarative_base()
# 1. 放入 DefectRecord 类
class DefectRecord(Base):
    __tablename__ = 'defects'
    id = Column(Integer, primary_key=True)
    filename = Column(String)
    timestamp = Column(DateTime, default=datetime.now)
    channel = Column(Integer)
    defect_type = Column(String)
    loc_x = Column(Integer)
    loc_y = Column(Integer)
    val = Column(Integer)  # 存亮度值
    pass

# 2. 放入数据库初始化代码
db_path = os.path.join(BASE_DIR, "defects_v5.db")
engine = create_engine(f"sqlite:///{db_path}")
Base.metadata.create_all(engine)
Session = sessionmaker(bind=engine)