from config.Config import Config
import pandas as pd


def read_csv(fileName):
	fileToRead = Config.DataPath + fileName
	data = pd.read_csv(fileToRead)
	return data
