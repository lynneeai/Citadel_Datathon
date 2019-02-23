import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from statsmodels.tsa.arima_model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.graphics import utils
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from sklearn import linear_model
from sklearn.metrics import r2_score

renewable_file = '../data/Renewable_Production.csv'

def read_csv():
	data_dict = defaultdict(list)
	with open(renewable_file, 'r') as csv_file:
		csv_reader = csv.reader(csv_file, delimiter=',')
		prev_energy = None
		current_energy = None
		energy_value = None
		energy_year = None
		for row in csv_reader:
			try:
				year = int(row[2])
				if row[1] == 'Biomass total consumption':
					current_energy = 'bio'
				elif row[1] == 'Geothermal energy total consumption':
					current_energy = 'geo'
				elif row[1] == 'Hydropower total consumption':
					current_energy = 'hydro'
				elif row[1] == 'Solar energy total consumption':
					current_energy = 'solar'
				elif row[1] == 'Wind energy total consumption':
					current_energy = 'wind'
				elif row[1] == 'Nuclear energy consumed for electricity generation, total':
					current_energy = 'nuclear'

				if current_energy != prev_energy:
					data_dict[prev_energy] = [np.array(energy_year), np.array(energy_value)]

					prev_energy = current_energy
					energy_value = []
					energy_year = []
				
				value = int(row[3])
				if value != 0:
					energy_value.append(value)
					energy_year.append(year)

			except Exception:
				continue
		# last energy
		data_dict[prev_energy] = [np.array(energy_year), np.array(energy_value)]
	return data_dict

def plot_data(years, data, title):
	plt.plot(years, data)
	plt.title(title)
	plt.show()

def plot_acfpac(data):
	plot_acf(data)
	plot_pacf(data)
	pyplot.show()

def arima_forecasting(actual, p, d, q, steps=1):
	model = ARIMA(actual, order=(p, d, q))
	model_fit = model.fit(disp=0)
	prediction = model_fit.forecast(steps)[0]
	return prediction

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def predict_future(prev_data, p, d, q, years=100):
	actual = prev_data.tolist()
	pred = arima_forecasting(actual, p, d, q, steps=years)
	# print(pred)
	return pred

def evaluate(d_train, d_val, p, d, q):
    tm=np.mean(d_train)
    tv=np.std(d_train)
    d_val=(d_val-tm)/tv
    actual = ((d_train-tm)/tv).tolist()
    predictions = []
    for time in range(len(d_val)):
        act = d_val[time]
        pred = arima_forecasting(actual, p, d, q)[0]
        actual.append(act)
        predictions.append(pred)
    error = rmse(np.array(predictions), d_val)
    print('rmse: {0}'.format(error))
    return error

def param_gridsearch(data_train, p_range, d_range, q_range):
	tscv = TimeSeriesSplit(n_splits=3)

	best_avg_rmse = float("inf")
	p_best = None
	d_best = None
	q_best = None

	for p in p_range:
		for d in d_range:
			for q in q_range:
				errors = []
				# cross validation
				for train_idx, val_idx in tscv.split(data_train):
					try:
						d_train, d_val = data_train[train_idx], data_train[val_idx]
						rmse = evaluate(d_train, d_val, p, d, q)
						errors.append(rmse)
					except Exception:
						print(len(d_train))
						continue
				avg_rmse = np.mean(np.array(errors))
				if avg_rmse < best_avg_rmse:
					best_avg_rmse = avg_rmse
					p_best = p
					d_best = d
					q_best = q
	print('best avg rmse: {0}'.format(best_avg_rmse))
	return (p_best, d_best, q_best)

def split_train_test(years, values, percentage):
	split = int(len(years) * percentage)
	train = values[:split]
	test = values[split:]
	return (train, test, split)

def write_csv(nuclear, geo, bio, hydro, wind, solar):
	# nuclear_year, nuclear_data = nuclear[0], nuclear[1]
	# geo_year, geo_data = geo[0], geo[1]
	# bio_year, bio_data = bio[0], bio[1]
	# hydro_year, hydro_data = hydro[0], hydro[1]
	# wind_year, wind_data = wind[0], wind[1]
	# solar_year, solar_data = solar[0], solar[1]
	with open('../data/renewable_predict.csv', 'w') as outfile:
		fieldnames = ['year', 'nuclear', 'geo', 'bio', 'hydro', 'wind', 'solar']
		writer = csv.DictWriter(outfile, fieldnames=fieldnames)
		writer.writeheader()
		for i, year in enumerate(range(1960, 2517)):
			if year < 2005:
				writer.writerow({'year':year, 'nuclear':nuclear[i], 'geo':geo[i], 'bio':bio[i], 'hydro':hydro[i], 'wind':0, 'solar':0})
			elif year < 2010:
				writer.writerow({'year':year, 'nuclear':nuclear[i], 'geo':geo[i], 'bio':bio[i], 'hydro':hydro[i], 'wind':wind[year-2005], 'solar':0})
			else:
				writer.writerow({'year':year, 'nuclear':nuclear[i], 'geo':geo[i], 'bio':bio[i], 'hydro':hydro[i], 'wind':wind[year-2005], 'solar':solar[year-2010]})

if __name__ == "__main__":
	data_dict = read_csv()

	nuclear_year = data_dict['nuclear'][0]
	nuclear_data = data_dict['nuclear'][1]
	geo_year = data_dict['geo'][0]
	geo_data = data_dict['geo'][1]
	bio_year = data_dict['bio'][0]
	bio_data = data_dict['bio'][1]
	hydro_year = data_dict['hydro'][0]
	hydro_data = data_dict['hydro'][1]
	solar_year = data_dict['solar'][0]
	solar_data = data_dict['solar'][1]
	wind_year = data_dict['wind'][0]
	wind_data = data_dict['wind'][1]

	### param tuning ###
	# plot_data(nuclear_year, nuclear_data, 'nuclear')
	# plot_acfpac(nuclear_data)

	# plot_data(geo_year, geo_data, 'geo')
	# plot_acfpac(geo_data)

	# plot_data(bio_year, bio_data, 'bio')
	# plot_acfpac(bio_data)

	# plot_data(hydro_year, hydro_data, 'hydro')
	# plot_acfpac(hydro_data)

	# plot_data(solar_year, solar_data, 'solar')
	# plot_data(solar_year[26:], solar_data[26:], 'solar')
	# plot_acfpac(np.log(solar_data[26:]))
	solar_year = solar_year[26:]
	solar_data = solar_data[26:]

	# plot_data(wind_year, wind_data, 'wind')
	# plot_data(wind_year[22:-1], wind_data[22:-1], 'wind')
	# plot_acfpac(wind_data[22:-1])
	wind_year = wind_year[22:-1]
	wind_data = wind_data[22:-1]

	### evaluation ###
	percentage = 0.7

	nuclear_train, nuclear_test, nuclear_split = split_train_test(nuclear_year, nuclear_data, percentage)
	geo_train, geo_test, geo_split = split_train_test(geo_year, geo_data, percentage)
	bio_train, bio_test, bio_split = split_train_test(bio_year, bio_data, percentage)
	hydro_train, hydro_test, hydro_split = split_train_test(hydro_year, hydro_data, percentage)
	wind_train, wind_test, wind_split = split_train_test(wind_year, wind_data, percentage)

	# evaluate(nuclear_train, nuclear_test, 3, 1, 0)
	# evaluate(geo_train, geo_test, 3, 1, 0)
	# evaluate(bio_train, bio_test, 2, 1, 0)
	# evaluate(np.log(solar_train), np.log(solar_test), 0, 1, 0)
	# evaluate(wind_train, wind_test, 1, 1, 0)

	# sm = np.mean(solar_data)
	# sstd = np.std(solar_data)
	# solar_normalized = (solar_data - sm) / sstd
	# solar_train, solar_test, solar_split = split_train_test(solar_year, solar_normalized, percentage)
	# regr_eval = linear_model.LinearRegression()
	# regr_eval.fit(solar_year[:solar_split].reshape(-1, 1), solar_train)
	# solar_pred_eval = regr_eval.predict(solar_year[solar_split:].reshape(-1, 1))
	# solar_score = rmse(solar_pred_eval, solar_test)
	# print(solar_score)
	
	### predict ###
	years = 500
	nuclear_preds = predict_future(nuclear_data, 3, 1, 0, years=years)
	nuclear = np.concatenate((nuclear_data, nuclear_preds), axis = 0)
	nuclear_year = np.concatenate((nuclear_year, range(2017, 2017+years)), axis=0)
	# plot_data(nuclear_year, nuclear, 'nuclear')

	geo_preds = predict_future(geo_data, 3, 1, 0, years=years)
	geo = np.concatenate((geo_data, geo_preds), axis = 0)
	geo_year = np.concatenate((geo_year, range(2017, 2017+years)), axis=0)
	# plot_data(geo_year, geo, 'geo')

	bio_preds = predict_future(bio_data, 2, 1, 0, years=years)
	bio = np.concatenate((bio_data, bio_preds), axis = 0)
	bio_year = np.concatenate((bio_year, range(2017, 2017+years)), axis=0)
	# plot_data(bio_year, bio, 'bio')

	hydro_preds = predict_future(hydro_data, 1, 1, 0, years=years)
	hydro = np.concatenate((hydro_data, hydro_preds), axis = 0)
	hydro_year = np.concatenate((hydro_year, range(2017, 2017+years)), axis=0)
	# plot_data(hydro_year, hydro, 'hydro')

	wind_preds = predict_future(wind_data, 1, 1, 1, years=years+1)
	wind = np.concatenate((wind_data, wind_preds), axis=0)
	wind_year = np.concatenate((wind_year, range(2016, 2016+years+1)), axis=0)
	# plot_data(wind_year, wind, 'wind')

	regr = linear_model.LinearRegression()
	regr.fit(solar_year.reshape(-1, 1), solar_data)
	solar_preds = regr.predict(np.array(range(2017, 2017+years)).reshape(-1, 1))
	solar = np.concatenate((solar_data, solar_preds), axis=0)
	solar_year = np.concatenate((solar_year, range(2017, 2017+years)), axis=0)
	# plot_data(solar_year, solar, 'solar')

	write_csv(nuclear, geo, bio, hydro, wind, solar)
