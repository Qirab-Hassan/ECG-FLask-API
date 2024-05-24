import aiohttp
from flask import Flask, jsonify, request
import numpy as np
from keras.models import load_model
import asyncio
from apscheduler.schedulers.background import BackgroundScheduler

app = Flask(__name__)

model = load_model('ECG_Model.h5')
api_key = "BBUS-llkCCBCGG4YJGJjsl12mywraAcfQkV"
latest_timestamp = None 
ECG_Values = [] 
terminate_flag = False  
communication_broken = False  
heartbeat_received = True  

async def fetch_latest_ECG_Values_from_ubidots(api_key, latest_timestamp, ECG_Values):
    live_count = 0
    url = 'https://industrial.api.ubidots.com/api/v1.6/devices/apneasense/ecg/values/?page_size=1'
    headers = {"X-Auth-Token": api_key}
    global terminate_flag, communication_broken

    async with aiohttp.ClientSession() as session:
        while not terminate_flag:  
            if communication_broken:
                ECG_Values.clear() 
                return None
            try:
                async with session.get(url, headers=headers) as response:
                    response.raise_for_status()
                    data = await response.json()

                    if "results" in data and data["results"]:
                        latest_entry = data["results"][0]
                        if latest_entry['timestamp'] != latest_timestamp:
                            ECG_Values.append(latest_entry['value'])
                            live_count += 1
                            print("Element Added :: ", live_count)
                            if len(ECG_Values) == 88:
                                temp = ECG_Values.copy()
                                ECG_Values.clear()
                                return temp
                            latest_timestamp = latest_entry['timestamp']
            except aiohttp.ClientError as e:
                print("Error fetching SpO2 values:", e)
                return None  
        terminate_flag = False    
        print('Loop Terminated')
        return False

@app.route('/ecgpredict', methods=['GET'])
async def predict():
    ECG_Values_fetched = await fetch_latest_ECG_Values_from_ubidots(api_key, latest_timestamp, ECG_Values)
    if ECG_Values_fetched is None:
        return jsonify({'message': 'Communication with the server is currently broken. Please try again later.'}), 503
    elif not ECG_Values_fetched:
        return jsonify({'message': 'Process Terminated.'}), 200 
       
    reshaped_values = np.array(ECG_Values_fetched).reshape((1, 88))
    mean_test = np.mean(reshaped_values)
    std_test = np.std(reshaped_values)
    ECG_test_normalized = (reshaped_values - mean_test) / std_test
    ECG_test_normalized = np.expand_dims(ECG_test_normalized, axis=2)
    y_pred = model.predict(ECG_test_normalized)
    predict_test = np.argmax(y_pred, axis=1)
    predict_test = predict_test.reshape(predict_test.shape[0], 1)
    confidence = np.max(y_pred, axis=1)[0]
    print("Confidence: ", confidence)
    array_as_list = predict_test.tolist()
    return jsonify({'array': array_as_list}), 200

@app.route('/ecgterminate', methods=['POST'])
def terminate():
    global terminate_flag, ECG_Values
    terminate_flag = True
    ECG_Values.clear() 
    return jsonify({'message': 'Operation Successful.'}), 200

@app.route('/heartbeat', methods=['POST'])
def heartbeat():
    print('Heart Beat ...')
    global heartbeat_received, communication_broken
    heartbeat_received = True
    communication_broken = False  
    return jsonify({'message': 'Heartbeat received.'}), 200

def check_heartbeat():
    global heartbeat_received, communication_broken
    if not heartbeat_received:
        communication_broken = True
        print("Communication with Flutter app is broken.")
    heartbeat_received = False

if __name__ == '__main__':
    scheduler = BackgroundScheduler()
    scheduler.add_job(func=check_heartbeat, trigger="interval", seconds=10)
    scheduler.start()
    
    app.run(host='0.0.0.0', port=5000)
