from flask import Flask, jsonify
from flask_cors import CORS
import datetime
import json
from apscheduler.schedulers.background import BackgroundScheduler
from flask_socketio import SocketIO
from statistics import mean

import pandas as pd
import numpy as np

from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

import eventlet
eventlet.monkey_patch(thread=True, time=True)


app = Flask(__name__)
app.config['SECRET_KEY'] = 'bangpakong'
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app=app, cors_allowed_origins='*',
                    logger=True, engineio_logger=True)
scheduler = BackgroundScheduler(executor='gevent')


report = {
    "temple": {
        "current": {},
        "next_24": [],
        "actual_vs_pred": []
    },
    "bangkhla": {
        "current": {},
        "next_24": [],
        "actual_vs_pred": []
    }
}


report_daily = {
    "temple": {
        "today": {},
        "prev": {},
        "current": {},
        "next_14": [],
        "actual_vs_pred": []
    },
    "bangkhla": {
        "today": {},
        "prev": {},
        "current": {},
        "next_14": [],
        "actual_vs_pred": []
    }
}


time_step_ec_input_hourly = 48
time_step_ec_next_hourly = 24
window_ec_bangkhla_hourly = []
window_ec_temple_hourly = []

time_step_ec_input_daily = 28
time_step_ec_next_daily = 14
window_ec_bangkhla_daily = []
window_ec_temple_daily = []


model_bk_hourly = load_model('bilstm-24-2022-batch-24.h5')
model_bk_hourly.compile(loss='mean_squared_error',
                        optimizer='adam', metrics=['mse'])

model_bk_daily = load_model('lstm-14-2022-batch-24.h5')
model_bk_daily.compile(loss='mean_squared_error',
                       optimizer='adam', metrics=['mse'])

df_bk = pd.read_csv("chol-bangkla-2022.csv", parse_dates=["datetime"])
df_bk = df_bk.sort_values(by=['datetime'])
data_bk = df_bk.filter(['ec'])
dataset_bk = data_bk.values

scaler_bk = MinMaxScaler(feature_range=(0, 1))
scaler_bk = scaler_bk.fit(dataset_bk)

df_temple = pd.read_csv("meter-uscm.csv", parse_dates=["datetime"])
df_temple = df_temple.sort_values(by=['datetime'])
data_temple = df_temple.filter(['ec'])
dataset_temple = data_temple.values

scaler_temple = MinMaxScaler(feature_range=(0, 1))
scaler_temple = scaler_temple.fit(dataset_temple)


def map_today(today):
    today_json = {
        "date": today.strftime('%a %d-%m-%Y'),
        "time": today.strftime('%H:00')
    }

    return today_json


def read_from_meter(time):
    date_file = time.strftime("%Y%m%d")
    time_file = int(time.strftime("%H%M"))

    try:
        text = open(r'C:\Users\Administrator\Documents\GET\ec-meter-mean\meter-mean-{}-{:04d}.json'.format(
            date_file, time_file), mode="r")
        data_from_meter = json.load(text)
        text.close()

        uscm = round(data_from_meter["mean"] * 1000, 2)
        gl = round(uscm * .55 / 1000, 2)

    except:
        return read_from_meter(time - datetime.timedelta(hours=1))

    data = {
        "date": time.strftime('%a %d-%m-%Y'),
        "time": time.strftime('%H:00'),
        "uscm": uscm,
        "gl": gl
    }

    return data


def read_from_bangkhla(time):
    date_file = time.strftime("%Y%m%d")
    time_file = int(time.strftime("%H"))

    try:
        text = open(r'C:\Users\Administrator\Documents\GET\cholpratan-bangkla\SP07-{}-{:02d}00.json'.format(
            date_file, time_file), mode="r")
        data_from_bangkhla = json.load(text)
        text.close()

        data = {
            "date": time.strftime('%a %d-%m-%Y'),
            "time": time.strftime('%H:00'),
            "uscm": data_from_bangkhla[0]["DataConductivity"],
            "gl": data_from_bangkhla[0]["Salinity"]
        }
    except:
        return read_from_bangkhla(time-datetime.timedelta(hours=1))

    return data


def read_ec_from_bangkhla(time, step=3):

    date_file = time.strftime("%Y%m%d")
    time_file = int(time.strftime("%H%M"))

    ec_bangkhla = None

    try:
        text = open(r"C:\Users\Administrator\Documents\GET\cholpratan-bangkla\SP07-{}-{:04d}.json".format(
            date_file, time_file), encoding='UTF-8', mode="r")
        data_from_bangkhla = json.load(text)
        text.close()

        ec_bangkhla = float(data_from_bangkhla[0]["DataConductivity"])

        return {"bangkhla": ec_bangkhla,
                "date": date_file,
                "time": "{:04d}".format(time_file),
                "original": True}
    except:
        if step >= 0:
            time -= datetime.timedelta(minutes=15)

            return {**read_ec_from_bangkhla(time, step-1), **{"original": False}}
        else:
            time -= datetime.timedelta(days=1)
            time += datetime.timedelta(hours=1)
            time.replace(minute=0)

            return {**read_ec_from_bangkhla(time), **{"original": False}}


def cal_ec_temple_mean(time):

    time_init = time

    time -= datetime.timedelta(minutes=1)

    data = {"mean": 0,
            "count": 0,
            "raw_data": [],
            "date_display": time_init.strftime("%Y%m%d"),
            "time_display": "{:04d}".format(int(time_init.strftime("%H%M")))}

    while 1:
        date_file = time.strftime("%Y%m%d")
        time_file = int(time.strftime("%H%M"))

        try:
            text = open(r"C:\Users\Administrator\Documents\GET\from-meter\meter-{}-{:04d}10.json".format(
                date_file, time_file), encoding='UTF-8', mode="r")
            data_from_temple = json.load(text)
            text.close()

            ec_temple = float(data_from_temple["Sensor"]["EC"])

            data_sub = {
                "temple": ec_temple,
                "date": date_file,
                "time": "{:04d}".format(time_file)
            }

            data["raw_data"].append(data_sub)

        except:
            pass

        if time.minute == 0:
            break

        time -= datetime.timedelta(minutes=1)

    temple_values = [element["temple"] for element in data["raw_data"]]
    data["mean"] = round(float(np.mean(temple_values)),
                         2) if len(temple_values) > 0 else None
    data["count"] = len(temple_values)

    date_file_output = time_init.strftime("%Y%m%d")
    time_file_output = int(time_init.strftime("%H%M"))
    with open(r"C:\Users\Administrator\Documents\GET\ec-meter-mean\meter-mean-{}-{:04d}.json".format(date_file_output, time_file_output),
              "w", encoding='UTF-8') as outfile:
        json.dump(data, outfile, indent=4, ensure_ascii=False)


def read_ec_from_temple(time):

    date_file = time.strftime("%Y%m%d")
    time_file = int(time.strftime("%H%M"))

    ec_temple = None

    try:
        text = open(r"C:\Users\Administrator\Documents\GET\ec-meter-mean\meter-mean-{}-{:04d}.json".format(
            date_file, time_file), encoding='UTF-8', mode="r")
        data_from_temple = json.load(text)
        text.close()

        ec_temple = data_from_temple["mean"]

        return {"temple": round(ec_temple * 1000, 2),
                "date": date_file,
                "time": "{:04d}".format(time_file),
                "original": True}
    except:
        time -= datetime.timedelta(days=1)

        return {**read_ec_from_temple(time), **{"original": False}}


def save_input(array, time, location, location_save, filename):
    date_file = time.strftime("%Y%m%d")
    time_file = int(time.strftime("%H00"))

    file_name = filename.format(date_file, time_file)

    report_ec_input = {
        "date": date_file,
        "time": "{:04d}".format(time_file),
        "input": array
    }
    with open(r"C:\Users\Administrator\Documents\GET\{}\{}.json".format(location_save, file_name), "w", encoding='UTF-8') as outfile:
        json.dump(report_ec_input, outfile, indent=4, ensure_ascii=False)


def predict_ec_hourly(array, scaler, time_step_next, time, location, location_save, filename):
    values = [i.get(location) for i in array]
    window = np.array(values, ndmin=2).T
    window_scaled = scaler.transform(window)
    window_scaled = np.reshape(
        window_scaled, (1, window_scaled.shape[0], window_scaled.shape[1]))

    window_pred = model_bk_hourly.predict(window_scaled)
    window_pred_inversed = scaler.inverse_transform(window_pred)

    timestamps = [
        time + i*datetime.timedelta(hours=1) for i in range(1, time_step_next + 1)]

    date_file = time.strftime("%Y%m%d")
    time_file = int(time.strftime("%H00"))

    file_name = filename.format(date_file, time_file)

    ec_next_24 = {
        "date": date_file,
        "time": time_file,
        "output": []
    }

    pred_list = [{
        "date": timestamps[i].strftime('%a %d-%m-%Y'),
        "time": timestamps[i].strftime('%H:00'),
        "gl": round(float(window_pred_inversed[0][i]) * .55 / 1000, 2),
        "uscm": round(float(window_pred_inversed[0][i]), 2)
    } for i in range(time_step_next)]

    ec_next_24["output"] = pred_list

    with open(r"C:\Users\Administrator\Documents\GET\ec-hourly-prediction\{}\{}.json".format(location_save, file_name), "w", encoding='UTF-8') as outfile:
        json.dump(ec_next_24, outfile, indent=4, ensure_ascii=False)

    return pred_list


def predict_ec_daily(array, scaler, time_step_next, time, location, location_save, filename):
    values = [i.get(location) for i in array]
    window = np.array(values, ndmin=2).T
    window_scaled = scaler.transform(window)
    window_scaled = np.reshape(
        window_scaled, (1, window_scaled.shape[0], window_scaled.shape[1]))

    window_pred = model_bk_daily.predict(window_scaled)
    window_pred_inversed = scaler.inverse_transform(window_pred)

    timestamps = [
        time + i*datetime.timedelta(days=1) for i in range(1, time_step_next + 1)]

    date_file = time.strftime("%Y%m%d")
    time_file = int(time.strftime("%H00"))

    file_name = filename.format(date_file, time_file)

    ec_next_14 = {
        "date": date_file,
        "time": time_file,
        "output": []
    }

    pred_list = [{
        "date": timestamps[i].strftime('%a %d-%m-%Y'),
        "time": timestamps[i].strftime('%H:00'),
        "gl": round(float(window_pred_inversed[0][i]) * .55 / 1000, 2),
        "uscm": round(float(window_pred_inversed[0][i]), 2)
    } for i in range(time_step_next)]

    ec_next_14["output"] = pred_list

    with open(r"C:\Users\Administrator\Documents\GET\ec-daily-prediction\{}\{}.json".format(location_save, file_name), "w", encoding='UTF-8') as outfile:
        json.dump(ec_next_14, outfile, indent=4, ensure_ascii=False)

    return pred_list


def read_history(time, location, filename, fn_to_read, var_to_read):
    date_file_pred = time.strftime("%Y%m%d")
    time_file = int(time.strftime("%H00"))

    file_name = filename.format(date_file_pred, time_file)

    try:
        text = open(r"C:\Users\Administrator\Documents\GET\{}\{}.json".format(
            location, file_name), encoding='UTF-8', mode="r")
        pred_from_file = json.load(text)
        text.close()

    except:
        return []

    array = [{
        "date": p["date"],
        "time": p["time"],
        "datetime": datetime.datetime.strptime(p["date"]+" "+p["time"], '%a %d-%m-%Y %H:%M').replace(tzinfo=None).strftime('%a %d-%m-%Y %H:%M'),
        "gl_pred": p["gl"],
        "gl_actual": fn_to_read(datetime.datetime.strptime(p["date"]+" "+p["time"], '%a %d-%m-%Y %H:%M'))[var_to_read]
    }
        for p in pred_from_file["output"]]

    return array


def generate_report():
    time_now = datetime.datetime.now()
    time_now = time_now.replace(minute=0)

    date_file = time_now.strftime("%Y%m%d")
    time_file = int(time_now.strftime("%H%M"))

    time_temple = time_now
    time_bangkhla = time_now

    try:
        open(r'C:\Users\Administrator\Documents\GET\ec-meter-mean\meter-mean-{}-{:04d}.json'.format(
            date_file, time_file), mode="r")
    except:
        time_temple -= datetime.timedelta(hours=1)

    try:
        open(r'C:\Users\Administrator\Documents\GET\cholpratan-bangkla\SP07-{}-{:04d}.json'.format(
            date_file, time_file), mode="r")
    except:
        time_bangkhla -= datetime.timedelta(hours=1)

    cal_ec_temple_mean(time_temple)

    report["temple"]["current"] = read_from_meter(time_now)
    report["bangkhla"]["current"] = read_from_bangkhla(time_now)

    for j in range(1, time_step_ec_input_hourly + 1):
        input_prev_bangkhla = (
            time_bangkhla - datetime.timedelta(hours=(j - 1)))
        input_prev_temple = (time_temple - datetime.timedelta(hours=(j - 1)))
        window_ec_bangkhla_hourly.insert(
            0, read_ec_from_bangkhla(input_prev_bangkhla))
        window_ec_temple_hourly.insert(
            0, read_ec_from_temple(input_prev_temple))

    save_input(window_ec_bangkhla_hourly, time_bangkhla, "bangkhla",
               "ec-hourly-prediction\\bangkhla\\input_48", "ec-input-48-{}-{:04d}")
    save_input(window_ec_temple_hourly, time_temple, "temple",
               "ec-hourly-prediction\\temple\\input_48", "ec-input-48-{}-{:04d}")

    time_temple_start = datetime.datetime(2023, 3, 2, 23, 0)
    time_bangkhla_start = datetime.datetime(2022, 12, 31, 23, 0)

    while 1:
        read_bk_output = read_history(
            time_bangkhla_start, "ec-hourly-prediction\\bangkhla\\next_24", "ec-next-24-{}-{:04d}", read_from_bangkhla, "gl")
        if (time_now - time_bangkhla_start).days < 1:
            break
        report["bangkhla"]["actual_vs_pred"].extend(read_bk_output)
        time_bangkhla_start += datetime.timedelta(days=1)

    if time_bangkhla.hour < 23:
        report["bangkhla"]["actual_vs_pred"].extend(
            read_history(time_bangkhla_start, "ec-hourly-prediction\\bangkhla\\next_24", "ec-next-24-{}-{:04d}", read_from_bangkhla, "gl")[:(time_bangkhla.hour + 1)])

    while 1:
        read_temple_output = read_history(
            time_temple_start, "ec-hourly-prediction\\temple\\next_24", "ec-next-24-{}-{:04d}", read_from_meter, "gl")
        if (time_now - time_temple_start).days < 1:
            break
        report["temple"]["actual_vs_pred"].extend(read_temple_output)
        time_temple_start += datetime.timedelta(days=1)

    if time_temple.hour < 23:
        report["temple"]["actual_vs_pred"].extend(
            read_history(time_temple_start, "ec-hourly-prediction\\temple\\next_24", "ec-next-24-{}-{:04d}", read_from_meter, "gl")[:(time_temple.hour + 1)])

    report["bangkhla"]["next_24"] = predict_ec_hourly(
        window_ec_bangkhla_hourly, scaler_bk, time_step_ec_next_hourly, time_bangkhla, "bangkhla", "bangkhla\\next_24", "ec-next-24-{}-{:04d}")
    report["temple"]["next_24"] = predict_ec_hourly(
        window_ec_temple_hourly, scaler_temple, time_step_ec_next_hourly, time_temple, "temple", "temple\\next_24", "ec-next-24-{}-{:04d}")


def generate_report_daily():
    time_now = datetime.datetime.now()
    time_now = time_now.replace(minute=0, hour=6)

    date_file = time_now.strftime("%Y%m%d")
    time_file = int(time_now.strftime("%H"))

    time_temple = time_now
    time_bangkhla = time_now

    time_temple_start = datetime.datetime(2023, 3, 28, 6, 0)
    time_bangkhla_start = datetime.datetime(2022, 12, 31, 6, 0)

    try:
        open(r'C:\Users\Administrator\Documents\GET\ec-meter-mean\meter-mean-{}-{:02d}00.json'.format(
            date_file, time_file), mode="r")
    except:
        time_temple -= datetime.timedelta(days=1)

    try:
        open(r'C:\Users\Administrator\Documents\GET\cholpratan-bangkla\SP07-{}-{:02d}00.json'.format(
            date_file, time_file), mode="r")
    except:
        time_bangkhla -= datetime.timedelta(days=1)

    report_daily["temple"]["current"] = read_from_meter(time_temple)
    report_daily["bangkhla"]["current"] = read_from_bangkhla(time_bangkhla)

    report_daily["temple"]["prev"] = read_from_meter(
        time_temple - datetime.timedelta(days=1))
    report_daily["bangkhla"]["prev"] = read_from_bangkhla(
        time_bangkhla - datetime.timedelta(days=1))

    report_daily["temple"]["today"] = map_today(time_temple)
    report_daily["bangkhla"]["today"] = map_today(time_bangkhla)

    for j in range(1, time_step_ec_input_daily + 1):
        input_prev_bangkhla = (
            time_bangkhla - datetime.timedelta(days=(j - 1)))
        input_prev_temple = (time_temple - datetime.timedelta(days=(j - 1)))
        window_ec_bangkhla_daily.insert(
            0, read_ec_from_bangkhla(input_prev_bangkhla))
        window_ec_temple_daily.insert(
            0, read_ec_from_temple(input_prev_temple))

    save_input(window_ec_bangkhla_daily, time_bangkhla, "bangkhla",
               "ec-daily-prediction\\bangkhla\\input_28", "ec-input-28-{}-{:04d}")
    save_input(window_ec_temple_daily, time_temple, "temple",
               "ec-daily-prediction\\temple\\input_28", "ec-input-28-{}-{:04d}")

    report_daily["bangkhla"]["next_14"] = predict_ec_daily(
        window_ec_bangkhla_daily, scaler_bk, time_step_ec_next_daily, time_bangkhla, "bangkhla", "bangkhla\\next_14", "ec-next-14-{}-{:04d}")
    report_daily["temple"]["next_14"] = predict_ec_daily(
        window_ec_temple_daily, scaler_temple, time_step_ec_next_daily, time_temple, "temple", "temple\\next_14", "ec-next-14-{}-{:04d}")

    while 1:
        report_daily["bangkhla"]["actual_vs_pred"].append(
            read_history(time_bangkhla_start, "ec-daily-prediction\\bangkhla\\next_14", "ec-next-14-{}-{:04d}", read_from_bangkhla, "gl")[0])

        time_bangkhla_start += datetime.timedelta(days=1)

        if (time_now - time_bangkhla_start).days < 1:
            break

    while 1:
        report_daily["temple"]["actual_vs_pred"].append(
            read_history(time_temple_start, "ec-daily-prediction\\temple\\next_14", "ec-next-14-{}-{:04d}", read_from_meter, "gl")[0])

        time_temple_start += datetime.timedelta(days=1)

        if (time_now - time_temple_start).days < 1:
            break


@ app.route('/', methods=["GET"])
def first_entrance():
    with app.app_context():
        return jsonify(report)


@ socketio.on("post_hourly", namespace="/hourly")
def post_hourly():
    with app.app_context():
        time_now = datetime.datetime.now()
        time_now = time_now.replace(minute=0)

        cal_ec_temple_mean(time_now)

        time_read = time_now - datetime.timedelta(days=1)
        time_read = time_read.replace(hour=23)

        report["temple"]["current"] = read_from_meter(time_now)
        report["bangkhla"]["current"] = read_from_bangkhla(time_now)

        window_ec_bangkhla_hourly.pop(0)
        window_ec_bangkhla_hourly.append(read_ec_from_bangkhla(time_now))

        window_ec_temple_hourly.pop(0)
        window_ec_temple_hourly.append(read_ec_from_temple(time_now))

        save_input(window_ec_bangkhla_hourly, time_now, "bangkhla",
                   "ec-hourly-prediction\\bangkhla\\input_48", "ec-input-48-{}-{:04d}")
        save_input(window_ec_temple_hourly, time_now, "temple",
                   "ec-hourly-prediction\\temple\\input_48", "ec-input-48-{}-{:04d}")

        report["bangkhla"]["next_24"] = predict_ec_hourly(
            window_ec_bangkhla_hourly, scaler_bk, time_step_ec_next_hourly, time_now, "bangkhla", "bangkhla\\next_24", "ec-next-24-{}-{:04d}")
        report["temple"]["next_24"] = predict_ec_hourly(
            window_ec_temple_hourly, scaler_temple, time_step_ec_next_hourly, time_now, "temple", "temple\\next_24", "ec-next-24-{}-{:04d}")

        report["bangkhla"]["actual_vs_pred"].append(
            read_history(time_read, "ec-hourly-prediction\\bangkhla\\next_24", "ec-next-24-{}-{:04d}", read_from_bangkhla, "gl")[time_now.hour])

        report["temple"]["actual_vs_pred"].append(
            read_history(time_read, "ec-hourly-prediction\\temple\\next_24", "ec-next-24-{}-{:04d}", read_from_meter, "gl")[time_now.hour])

        data_json = json.dumps(report)
        socketio.emit("post_hourly", data_json,
                      broadcast=True, namespace="/hourly")


@ app.route('/daily', methods=["GET"])
def first_entrance_daily():
    with app.app_context():
        return jsonify(report_daily)


@ socketio.on("post_daily", namespace="/daily")
def post_daily():
    with app.app_context():
        time_now = datetime.datetime.now()
        time_now = time_now.replace(minute=0, hour=6)

        time_read_bk = time_now - datetime.timedelta(days=1)
        time_read_temple = time_now - datetime.timedelta(days=1)

        report_daily["temple"]["prev"] = report_daily["temple"]["current"]
        report_daily["bangkhla"]["prev"] = report_daily["bangkhla"]["current"]

        report_daily["temple"]["current"] = read_from_meter(time_now)
        report_daily["bangkhla"]["current"] = read_from_bangkhla(time_now)

        report_daily["temple"]["today"] = map_today(time_now)
        report_daily["bangkhla"]["today"] = map_today(time_now)

        window_ec_bangkhla_daily.pop(0)
        window_ec_bangkhla_daily.append(read_ec_from_bangkhla(time_now))

        window_ec_temple_daily.pop(0)
        window_ec_temple_daily.append(read_ec_from_temple(time_now))

        save_input(window_ec_bangkhla_daily, time_now, "bangkhla",
                   "ec-daily-prediction\\bangkhla\\input_28", "ec-input-28-{}-{:04d}")
        save_input(window_ec_temple_daily, time_now, "temple",
                   "ec-daily-prediction\\temple\\input_28", "ec-input-28-{}-{:04d}")

        report_daily["bangkhla"]["next_14"] = predict_ec_daily(
            window_ec_bangkhla_daily, scaler_bk, time_step_ec_next_daily, time_now, "bangkhla", "bangkhla\\next_14", "ec-next-14-{}-{:04d}")
        report_daily["temple"]["next_14"] = predict_ec_daily(
            window_ec_temple_daily, scaler_temple, time_step_ec_next_daily, time_now, "temple", "temple\\next_14", "ec-next-14-{}-{:04d}")

        report_daily["bangkhla"]["actual_vs_pred"].append(
            read_history(time_read_bk, "ec-daily-prediction\\bangkhla\\next_14", "ec-next-14-{}-{:04d}", read_from_bangkhla, "gl")[0])

        report_daily["temple"]["actual_vs_pred"].append(
            read_history(time_read_temple, "ec-daily-prediction\\temple\\next_14", "ec-next-14-{}-{:04d}", read_from_meter, "gl")[0])

        data_json = json.dumps(report_daily)
        socketio.emit("post_daily", data_json,
                      broadcast=True, namespace="/daily")


if __name__ == '__main__':
    with app.app_context():
        generate_report()
        generate_report_daily()

        scheduler.add_job(post_hourly, "cron", hour="*",
                          minute="6", second="35")
        scheduler.add_job(post_daily, "cron", hour="6",
                          minute="6", second="35")
        scheduler.start()
        socketio.run(app, debug=True, port=20001, host="0.0.0.0")
