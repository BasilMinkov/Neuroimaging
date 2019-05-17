import pandas as pd

class TimeSeries(pd.DataFrame):

	def __init__(self, sampling_rate):
		pd.DataFrame.__init__(self)
		self.sampling_rate = sampling_rate