
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pickle
import json
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class model_input(BaseModel):
  a1:float
  a2:float
  a3:float
  a4:float
  a5:float
  a6:float
  a7:float
  a8:float
  a9:float
  a10:float
  a11:float
  a12:float
  a13:float
  a14:float
  a15:float
  a16:float
  a17:float
  a18:float
  a19:float
  a20:float
  a21:float
  a22:float
  a23:float
  a24:float
  a25:float
  a26:float
  a27:float
  a28:float
  a29:float
  a30:float
  a31:float
  a32:float
  a33:float
  a34:float
  a35:float
  a36:float
  a37:float
  a38:float
  a39:float
  a40:float
  a41:float
  a42:float
  a43:float
  a44:float
  a45:float
  a46:float
  a47:float
  a48:float
  a49:float
  a50:float
  a51:float
  a52:float
  a53:float
  a54:float
  a55:float
  a56:float
  a57:float
  a58:float
  a59:float
  a60:float
  a61:float
  a62:float
  a63:float
  a64:float
  a65:float
  a66:float
  a67:float
  a68:float
  a69:float
  a70:float
  a71:float
  a72:float
  a73:float
  a74:float
  a75:float
  a76:float
  a77:float
  a78:float
  a79:float
  a80:float
  a81:float
  a82:float
  a83:float
  a84:float
  a85:float
  a86:float
  a87:float
  a88:float
  a89:float
  a90:float
  a91:float
  a92:float
  a93:float
  a94:float
  a95:float
  a96:float
  a97:float
  a98:float
  a99:float
  a100:float
  a101:float
  a102:float
  a103:float
  a104:float
  a105:float
  a106:float
  a107:float
  a108:float
  a109:float
  a110:float
  a111:float
  a112:float
  a113:float
  a114:float
  a115:float
  a116:float
  a117:float
  a118:float
  a119:float
  a120:float
  a121:float
  a122:float
  a123:float
  a124:float
  a125:float
  a126:float
  a127:float
  a128:float
  a129:float
  a130:float
  a131:float
  a132:float
  a133:float
  a134:float
  a135:float
  a136:float
  a137:float
  a138:float
  a139:float
  a140:float

model=pickle.load(open('model.pkl','rb'))

@app.post('/ha_prediction')
def predicts(input_parameters:model_input):
  input_data=input_parameters.json()
  input_dic=json.loads(input_data)
  Novel_datapd=pd.DataFrame.from_dict(input_dic,orient='index')
  ND=Novel_datapd.transpose()
  ND=(ND+6.2808752)/13.6829783
  ND=tf.cast(ND,tf.float32)
  reconstructions = model(ND)
  threshold=0.027195456437766552
  loss = tf.keras.losses.mae(reconstructions, ND)
  return bool(tf.math.less(loss, threshold))









